"""
Final LangGraph chatbot backend.

What it provides:
  - async graph (runs on a dedicated background event loop) with AsyncSqliteSaver persistence
  - tools: web search, stock price (yfinance), calculator
  - PDF question answering (RAG), one PDF per chat thread
  - optional MCP tools (see MCP_SERVERS)
  - short chat titles, written by the model from the first message and saved in the database

Settings come from .env: OPENAI_API_KEY (required), OPENAI_MODEL (optional).
"""

import asyncio
import os
import tempfile
import threading
from typing import Annotated, Any, Dict, Optional, TypedDict

import aiosqlite
import yfinance as yf
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# Dedicated async loop for backend tasks
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()


def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop."""
    return _submit_async(coro)


# -------------------
# 1. LLM + embeddings
# -------------------
llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# -------------------
# 2. PDF retriever store (per thread)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a retriever for the uploaded PDF and store it for the thread.
    A newer upload replaces the previous PDF of that thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        name = filename or os.path.basename(temp_path)

        docs = PyPDFLoader(temp_path).load()
        for doc in docs:
            doc.metadata["source"] = name  # show the file name, not the temp path

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)
        if not chunks:
            raise ValueError(
                "No readable text found in this PDF. It may be a scanned or image-only file."
            )

        vector_store = InMemoryVectorStore.from_documents(chunks, embeddings)
        _THREAD_RETRIEVERS[str(thread_id)] = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )
        summary = {"filename": name, "documents": len(docs), "chunks": len(chunks)}
        _THREAD_METADATA[str(thread_id)] = summary
        return dict(summary)
    finally:
        # The vector store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return dict(_THREAD_METADATA.get(str(thread_id), {}))


# -------------------
# 3. Tools
# -------------------
class RetryingSearch(DuckDuckGoSearchRun):
    """DuckDuckGo search that tries once more if it fails (its backends are flaky)."""

    def _run(self, query: str, run_manager=None) -> str:
        try:
            return super()._run(query, run_manager=run_manager)
        except Exception:
            return super()._run(query, run_manager=run_manager)


search_tool = RetryingSearch(region="us-en")


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') from
    Yahoo Finance using yfinance. For non-US exchanges add the Yahoo suffix,
    e.g. 'RELIANCE.NS' for NSE. No API key required.
    """
    try:
        info = yf.Ticker(symbol).fast_info
        price = float(info["lastPrice"])
        previous_close = float(info["previousClose"])
        currency = info["currency"]
        if not price > 0:
            raise ValueError("no price available")
    except Exception:
        return {"error": f"Could not fetch a price for '{symbol}'. Check the ticker symbol or try again later."}

    change = price - previous_close
    return {
        "symbol": symbol.upper(),
        "price": round(price, 2),
        "previous_close": round(previous_close, 2),
        "change": round(change, 2),
        "change_percent": round(change / previous_close * 100, 2) if previous_close else None,
        "currency": currency,
    }


@tool
async def rag_tool(query: str, config: RunnableConfig) -> dict:
    """
    Retrieve relevant information from the PDF uploaded to this chat.
    """
    # The thread id comes from the run config, not from the model.
    thread_id = str(config.get("configurable", {}).get("thread_id"))
    retriever = _THREAD_RETRIEVERS.get(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = await retriever.ainvoke(query)
    return {
        "query": query,
        "context": [doc.page_content for doc in result],
        "metadata": [doc.metadata for doc in result],
        "source_file": _THREAD_METADATA.get(thread_id, {}).get("filename"),
    }


# MCP servers to load tools from. Each one is loaded on its own, so a broken server
# is skipped (with a message in the terminal) instead of disabling all MCP tools.
# Leave this empty to run without MCP tools. Example entries:
MCP_SERVERS = {
    # "math": {
    #     "transport": "stdio",
    #     "command": "python3",
    #     "args": ["/path/to/your/mcp-math-server/main.py"],
    # },
    # "remote": {
    #     "transport": "streamable_http",  # if this fails, try "sse"
    #     "url": "https://<your-server>/mcp",
    # },
}


def load_mcp_tools() -> list[BaseTool]:
    loaded = []
    for name, config in MCP_SERVERS.items():
        try:
            server_tools = run_async(MultiServerMCPClient({name: config}).get_tools())
        except Exception as exc:
            # MCP connection errors usually arrive wrapped in (nested) ExceptionGroups
            reason = exc
            while getattr(reason, "exceptions", None):
                reason = reason.exceptions[0]
            print(f"[MCP] Skipping server '{name}': {type(reason).__name__}: {reason}")
            continue
        print(f"[MCP] Loaded {len(server_tools)} tool(s) from '{name}'")
        loaded.extend(server_tools)
    return loaded


mcp_tools = load_mcp_tools()

tools = [search_tool, get_stock_price, calculator, rag_tool, *mcp_tools]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 4. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 5. Nodes
# -------------------
SYSTEM_PROMPT = (
    "You are a helpful assistant. For questions about the uploaded PDF, call the "
    "`rag_tool`. You can also use the web search, stock price, and calculator tools "
    "when helpful. If no document is available, ask the user to upload a PDF."
)


def _close_dangling_tool_calls(messages: list[BaseMessage]) -> list[BaseMessage]:
    """
    The model API rejects a tool call that has no result message. That can be left behind
    when a run is cut off between the call and its result, so give such a call an error
    result. Only the copy sent to the model changes, not the saved chat.
    """
    fixed = []
    for i, message in enumerate(messages):
        fixed.append(message)
        if isinstance(message, AIMessage) and message.tool_calls:
            answered = set()
            for later in messages[i + 1:]:
                if not isinstance(later, ToolMessage):
                    break
                answered.add(later.tool_call_id)
            for call in message.tool_calls:
                if call["id"] not in answered:
                    fixed.append(
                        ToolMessage(
                            content="Error: this tool call did not finish, so it has no result.",
                            tool_call_id=call["id"],
                            name=call["name"],
                            status="error",
                        )
                    )
    return fixed


async def chat_node(state: ChatState, config: RunnableConfig):
    """LLM node that may answer or request a tool call."""
    messages = [SystemMessage(content=SYSTEM_PROMPT), *_close_dangling_tool_calls(state["messages"])]
    response = await llm_with_tools.ainvoke(messages, config=config)
    return {"messages": [response]}


# A tool that raises becomes an error message the model can read and react to,
# instead of aborting the whole run.
tool_node = ToolNode(tools, handle_tool_errors=True)

# -------------------
# 6. Checkpointer
# -------------------
# Anchored to this file's folder so every launch location uses the same database.
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chatbot.db")


async def _init_checkpointer():
    conn = await aiosqlite.connect(database=DB_PATH)
    return AsyncSqliteSaver(conn)


checkpointer = run_async(_init_checkpointer())

# -------------------
# 7. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 8. Helpers
# -------------------
async def _alist_threads():
    threads = {}
    async for checkpoint in checkpointer.alist(None):  # newest checkpoint first
        threads.setdefault(checkpoint.config["configurable"]["thread_id"])
    return list(threads)


def retrieve_all_threads() -> list[str]:
    """All saved thread ids, most recently used first."""
    return run_async(_alist_threads())


def load_messages(thread_id: str) -> list[BaseMessage]:
    """Saved messages of a thread (empty list for a thread with no messages)."""
    state = run_async(chatbot.aget_state({"configurable": {"thread_id": thread_id}}))
    return state.values.get("messages", [])


# -------------------
# 9. Chat titles
# -------------------
# Every chat gets a short name, written by the model from the chat's first message and
# saved in the same database (table `thread_titles`). Chats that existed before titles
# were added are named after their first message.
TITLE_PROMPT = (
    "Write a short title (at most 5 words) for a chat that starts with the user message "
    "below. Reply with the title only: no quotes, no trailing punctuation."
)

_TITLES: Dict[str, str] = {}


def _fallback_title(text: str, limit: int = 40) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


async def _init_titles():
    conn = await aiosqlite.connect(database=DB_PATH)
    await conn.execute(
        "CREATE TABLE IF NOT EXISTS thread_titles (thread_id TEXT PRIMARY KEY, title TEXT NOT NULL)"
    )
    await conn.commit()
    async with conn.execute("SELECT thread_id, title FROM thread_titles") as cursor:
        _TITLES.update({thread_id: title for thread_id, title in await cursor.fetchall()})
    return conn


_titles_conn = run_async(_init_titles())


async def _save_title(thread_id: str, title: str):
    await _titles_conn.execute(
        "INSERT OR REPLACE INTO thread_titles (thread_id, title) VALUES (?, ?)",
        (thread_id, title),
    )
    await _titles_conn.commit()
    _TITLES[thread_id] = title


async def _name_chat(thread_id: str, first_message: str) -> str:
    title = _fallback_title(first_message) or "New chat"
    try:
        reply = await llm.ainvoke(
            [SystemMessage(content=TITLE_PROMPT), HumanMessage(content=first_message[:1000])],
            config={"run_name": "chat_title"},
        )
        title = " ".join(reply.text.split()).strip("\"'`.")[:50] or title
    except Exception as exc:
        print(f"[titles] Could not generate a title, using the first message: {exc}")
    await _save_title(thread_id, title)
    return title


async def _backfill_titles():
    """Name chats that were created before titles existed after their first message."""
    for thread_id in await _alist_threads():
        if thread_id in _TITLES:
            continue
        state = await chatbot.aget_state({"configurable": {"thread_id": thread_id}})
        first = next(
            (m for m in state.values.get("messages", []) if isinstance(m, HumanMessage) and m.text),
            None,
        )
        if first is not None:
            await _save_title(thread_id, _fallback_title(first.text) or "Untitled chat")


run_async(_backfill_titles())


def get_thread_title(thread_id: str) -> Optional[str]:
    """The chat's name, or None for a chat that has no messages yet."""
    return _TITLES.get(thread_id)


def request_title(thread_id: str, first_message: str):
    """
    Name a new chat in the background and return a future for the finished title.
    Until the model's title is ready, the first message stands in for it.
    """
    _TITLES[thread_id] = _fallback_title(first_message) or "New chat"
    return submit_async_task(_name_chat(thread_id, first_message))
