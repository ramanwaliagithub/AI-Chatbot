"""
Postgres + pgvector version of the chatbot backend.

The same features as the SQLite version in the parent folder, but all storage lives in one
Postgres database:
  - chat history: LangGraph's AsyncPostgresSaver
  - the chat list and chat names: table `chats`
  - PDFs: tables `pdf_documents` and `pdf_chunks` (each chunk's text and a pgvector embedding)
Every row belongs to a `user_id` and every query is filtered by it, so users never see each
other's chats or PDFs. The graph also refuses to run a chat that belongs to someone else.

Settings come from .env: DATABASE_URL and OPENAI_API_KEY (required), OPENAI_MODEL (optional).
"""

import asyncio
import json
import os
import tempfile
import threading
from typing import Annotated, Any, Dict, Optional, Tuple, TypedDict

import psycopg
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
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pgvector import Vector
from pgvector.psycopg import register_vector_async
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. Copy .env.example to .env, then start the local "
        "database with `docker compose up -d`."
    )

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
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536  # the size of that model's vectors; the pdf_chunks column must match
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

# -------------------
# 2. Database
# -------------------
# `chats` is the list of every user's chats. Deleting a chat row also deletes its PDF rows
# (ON DELETE CASCADE). The chat history itself lives in LangGraph's own tables.
SCHEMA = [
    "CREATE EXTENSION IF NOT EXISTS vector",
    """
    CREATE TABLE IF NOT EXISTS chats (
        thread_id  TEXT PRIMARY KEY,
        user_id    TEXT NOT NULL,
        title      TEXT NOT NULL DEFAULT 'New chat',
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS chats_user_recent ON chats (user_id, updated_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS pdf_documents (
        thread_id       TEXT PRIMARY KEY REFERENCES chats (thread_id) ON DELETE CASCADE,
        user_id         TEXT NOT NULL,
        filename        TEXT NOT NULL,
        documents       INTEGER NOT NULL,
        chunks          INTEGER NOT NULL,
        embedding_model TEXT NOT NULL
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS pdf_chunks (
        thread_id   TEXT NOT NULL REFERENCES chats (thread_id) ON DELETE CASCADE,
        user_id     TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        content     TEXT NOT NULL,
        metadata    JSONB NOT NULL,
        embedding   vector({EMBEDDING_DIMENSIONS}) NOT NULL,
        PRIMARY KEY (thread_id, chunk_index)
    )
    """,
]


class OwnedPostgresSaver(AsyncPostgresSaver):
    """
    A history store that only reads and writes a chat's history for the user who owns it.
    LangGraph saves a run's input before any node runs, so the check has to live here, in
    the storage layer, to stop one user's input from landing in another user's chat.
    """

    async def aget_tuple(self, config):
        user_id, thread_id = _identity(config)
        await _verify_owner(user_id, thread_id, create=False)
        return await super().aget_tuple(config)

    async def aput(self, config, checkpoint, metadata, new_versions):
        user_id, thread_id = _identity(config)
        await _verify_owner(user_id, thread_id, create=True)
        return await super().aput(config, checkpoint, metadata, new_versions)

    async def aput_writes(self, config, writes, task_id, task_path=""):
        user_id, thread_id = _identity(config)
        await _verify_owner(user_id, thread_id, create=True)
        return await super().aput_writes(config, writes, task_id, task_path)


async def _configure_connection(conn):
    await register_vector_async(conn)  # lets psycopg send and receive pgvector values


async def _init_database():
    # The extension and tables come first, on a plain connection: the pool's connections
    # register the vector type, which only exists once the extension is installed.
    async with await psycopg.AsyncConnection.connect(DATABASE_URL, autocommit=True) as conn:
        for statement in SCHEMA:
            await conn.execute(statement)
    pool = AsyncConnectionPool(
        DATABASE_URL,
        min_size=1,
        max_size=10,
        open=False,
        configure=_configure_connection,
        # What AsyncPostgresSaver requires of its connections
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    await pool.open(wait=True)
    saver = OwnedPostgresSaver(pool)
    await saver.setup()  # creates LangGraph's checkpoint tables the first time
    return pool, saver


_pool, checkpointer = run_async(_init_database())


async def _fetchone(sql: str, params=()):
    async with _pool.connection() as conn:
        cursor = await conn.execute(sql, params)
        return await cursor.fetchone()


async def _fetchall(sql: str, params=()):
    async with _pool.connection() as conn:
        cursor = await conn.execute(sql, params)
        return await cursor.fetchall()


async def _execute(sql: str, params=()) -> int:
    async with _pool.connection() as conn:
        cursor = await conn.execute(sql, params)
        return cursor.rowcount


# Small caches so the page does not query the database for every name on every rerun.
_TITLES: Dict[Tuple[str, str], str] = {}
_DOCS: Dict[Tuple[str, str], dict] = {}
_OWNED: set = set()  # (user_id, thread_id) pairs already confirmed against the database


async def _ensure_chat(user_id: str, thread_id: str) -> str:
    """Create the chat if it is new, and refuse it if it belongs to another user. Returns its title."""
    async with _pool.connection() as conn:
        await conn.execute(
            "INSERT INTO chats (thread_id, user_id) VALUES (%s, %s) ON CONFLICT (thread_id) DO NOTHING",
            (thread_id, user_id),
        )
        cursor = await conn.execute(
            "SELECT user_id, title FROM chats WHERE thread_id = %s", (thread_id,)
        )
        row = await cursor.fetchone()
    if row["user_id"] != user_id:
        raise PermissionError("This chat belongs to another user.")
    _TITLES[(user_id, thread_id)] = row["title"]
    _OWNED.add((user_id, thread_id))
    return row["title"]


def _identity(config: RunnableConfig) -> Tuple[str, str]:
    """The (user_id, thread_id) of a run. Both must be given, so nothing runs without an owner."""
    configurable = config.get("configurable", {})
    user_id, thread_id = configurable.get("user_id"), configurable.get("thread_id")
    if not user_id or not thread_id:
        raise ValueError("A run needs both a user_id and a thread_id in its config.")
    return str(user_id), str(thread_id)


async def _verify_owner(user_id: str, thread_id: str, create: bool) -> None:
    """
    Raise PermissionError if the chat belongs to someone else. A write (create=True) also
    creates the chat for this user if it is new; a read only checks and never creates.
    """
    if (user_id, thread_id) in _OWNED:
        return
    if create:
        await _ensure_chat(user_id, thread_id)
        return
    row = await _fetchone("SELECT user_id FROM chats WHERE thread_id = %s", (thread_id,))
    if row is not None:
        if row["user_id"] != user_id:
            raise PermissionError("This chat belongs to another user.")
        _OWNED.add((user_id, thread_id))


# -------------------
# 3. PDFs (one per chat, searched with pgvector)
# -------------------
def ingest_pdf(user_id: str, file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Read the uploaded PDF, embed its chunks and save them for this user's chat.
    A newer upload replaces the previous PDF of that chat.

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

        texts = [chunk.page_content for chunk in chunks]
        metadatas = [json.loads(json.dumps(chunk.metadata, default=str)) for chunk in chunks]
        vectors = embeddings.embed_documents(texts)
        summary = {"filename": name, "documents": len(docs), "chunks": len(chunks)}

        run_async(_save_pdf(user_id, thread_id, summary, texts, metadatas, vectors))
        return dict(summary)
    finally:
        # The chunks are saved in the database, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass


async def _save_pdf(user_id, thread_id, summary, texts, metadatas, vectors):
    await _ensure_chat(user_id, thread_id)  # also refuses a chat that belongs to someone else
    rows = [
        (thread_id, user_id, i, text, Jsonb(metadata), Vector(vector))
        for i, (text, metadata, vector) in enumerate(zip(texts, metadatas, vectors))
    ]
    async with _pool.connection() as conn:
        async with conn.transaction():  # all or nothing: a failure keeps the previous PDF
            await conn.execute("DELETE FROM pdf_chunks WHERE thread_id = %s", (thread_id,))
            await conn.execute(
                "INSERT INTO pdf_documents (thread_id, user_id, filename, documents, chunks, embedding_model) "
                "VALUES (%s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (thread_id) DO UPDATE SET filename = EXCLUDED.filename, "
                "documents = EXCLUDED.documents, chunks = EXCLUDED.chunks, "
                "embedding_model = EXCLUDED.embedding_model",
                (thread_id, user_id, summary["filename"], summary["documents"], summary["chunks"], EMBEDDING_MODEL),
            )
            async with conn.cursor() as cursor:
                await cursor.executemany(
                    "INSERT INTO pdf_chunks (thread_id, user_id, chunk_index, content, metadata, embedding) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    rows,
                )
    _DOCS[(user_id, thread_id)] = dict(summary)


async def _aget_doc(user_id: str, thread_id: str) -> Optional[dict]:
    """The chat's PDF details, or None. A PDF embedded with a different model is ignored."""
    key = (user_id, thread_id)
    if key not in _DOCS:
        row = await _fetchone(
            "SELECT filename, documents, chunks FROM pdf_documents "
            "WHERE thread_id = %s AND user_id = %s AND embedding_model = %s",
            (thread_id, user_id, EMBEDDING_MODEL),
        )
        if row is None:
            return None
        _DOCS[key] = dict(row)
    return _DOCS[key]


def thread_document_metadata(user_id: str, thread_id: str) -> dict:
    return dict(run_async(_aget_doc(user_id, thread_id)) or {})


def thread_has_document(user_id: str, thread_id: str) -> bool:
    return bool(thread_document_metadata(user_id, thread_id))


# -------------------
# 4. Tools
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
    Search the PDF the user uploaded to this chat. Use this first for any question the
    uploaded document might answer, even when the user does not mention the PDF.
    """
    # The user and chat come from the run config, not from the model.
    user_id, thread_id = _identity(config)
    doc = await _aget_doc(user_id, thread_id)
    if doc is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    query_vector = await embeddings.aembed_query(query)
    # pgvector: "<=>" is cosine distance, so the closest chunks come first.
    rows = await _fetchall(
        "SELECT content, metadata FROM pdf_chunks WHERE user_id = %s AND thread_id = %s "
        "ORDER BY embedding <=> %s LIMIT 4",
        (user_id, thread_id, Vector(query_vector)),
    )
    return {
        "query": query,
        "context": [row["content"] for row in rows],
        "metadata": [row["metadata"] for row in rows],
        "source_file": doc["filename"],
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
# 5. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 6. Nodes
# -------------------
BASE_PROMPT = (
    "You are a helpful assistant. You can search the web, look up stock prices and do "
    "calculations when that helps."
)


def _system_prompt(doc: Optional[dict]) -> str:
    """The model's instructions, which depend on whether this chat has a PDF attached."""
    if doc:
        return (
            f"{BASE_PROMPT} The user has uploaded a PDF to this chat: '{doc['filename']}' "
            f"({doc['documents']} pages). For any question the document could answer, even if "
            "the user does not mention it, call `rag_tool` first and answer from what it "
            "returns. Search the web only if the document does not cover the question, and "
            "say which source your answer came from."
        )
    return (
        f"{BASE_PROMPT} No PDF has been uploaded to this chat; if the user asks about a "
        "document, ask them to upload a PDF."
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
    user_id, thread_id = _identity(config)  # ownership was already enforced by the history store
    await _execute(
        "UPDATE chats SET updated_at = now() WHERE thread_id = %s AND user_id = %s",
        (thread_id, user_id),
    )  # recently used chats come first in the list
    doc = await _aget_doc(user_id, thread_id)
    messages = [
        SystemMessage(content=_system_prompt(doc)),
        *_close_dangling_tool_calls(state["messages"]),
    ]
    response = await llm_with_tools.ainvoke(messages, config=config)
    return {"messages": [response]}


# A tool that raises becomes an error message the model can read and react to,
# instead of aborting the whole run.
tool_node = ToolNode(tools, handle_tool_errors=True)

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
# 8. Chats: list, open, name, delete (always for one user)
# -------------------
def retrieve_all_threads(user_id: str) -> list[str]:
    """This user's chat ids, most recently used first."""
    rows = run_async(
        _fetchall(
            "SELECT thread_id, title FROM chats WHERE user_id = %s ORDER BY updated_at DESC",
            (user_id,),
        )
    )
    for row in rows:
        _TITLES[(user_id, row["thread_id"])] = row["title"]
        _OWNED.add((user_id, row["thread_id"]))
    return [row["thread_id"] for row in rows]


def load_messages(user_id: str, thread_id: str) -> list[BaseMessage]:
    """Saved messages of one of this user's chats (empty for any other chat)."""

    async def _load():
        owned = await _fetchone(
            "SELECT 1 AS ok FROM chats WHERE thread_id = %s AND user_id = %s", (thread_id, user_id)
        )
        if owned is None:
            return []
        state = await chatbot.aget_state(
            {"configurable": {"thread_id": thread_id, "user_id": user_id}}
        )
        return state.values.get("messages", [])

    return run_async(_load())


TITLE_PROMPT = (
    "Write a short title (at most 5 words) for a chat that starts with the user message "
    "below. Reply with the title only: no quotes, no trailing punctuation."
)


def _fallback_title(text: str, limit: int = 40) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


def get_thread_title(user_id: str, thread_id: str) -> Optional[str]:
    """The chat's name, or None for a chat that has not been saved yet."""
    return _TITLES.get((user_id, thread_id))


async def _name_chat(user_id: str, thread_id: str, first_message: str) -> str:
    await _ensure_chat(user_id, thread_id)
    title = _fallback_title(first_message) or "New chat"
    try:
        reply = await llm.ainvoke(
            [SystemMessage(content=TITLE_PROMPT), HumanMessage(content=first_message[:1000])],
            config={"run_name": "chat_title"},
        )
        title = " ".join(reply.text.split()).strip("\"'`.")[:50] or title
    except Exception as exc:
        print(f"[titles] Could not generate a title, using the first message: {exc}")
    await _execute(
        "UPDATE chats SET title = %s, updated_at = now() WHERE thread_id = %s AND user_id = %s",
        (title, thread_id, user_id),
    )
    _TITLES[(user_id, thread_id)] = title
    return title


def request_title(user_id: str, thread_id: str, first_message: str):
    """
    Name a chat in the background and return a future for the finished title.
    Until the model's title is ready, the first message stands in for it.
    """
    _TITLES[(user_id, thread_id)] = _fallback_title(first_message) or "New chat"
    return submit_async_task(_name_chat(user_id, thread_id, first_message))


async def _adelete_chat(user_id: str, thread_id: str) -> bool:
    # Only the owner can delete. Deleting the row also deletes the chat's PDF rows (cascade).
    deleted = await _execute(
        "DELETE FROM chats WHERE thread_id = %s AND user_id = %s", (thread_id, user_id)
    )
    if deleted:
        await checkpointer.adelete_thread(thread_id)  # the saved messages
    return bool(deleted)


def delete_chat(user_id: str, thread_id: str) -> bool:
    """Permanently remove one of this user's chats: its messages, name and saved PDF."""
    deleted = run_async(_adelete_chat(user_id, thread_id))
    _TITLES.pop((user_id, thread_id), None)
    _DOCS.pop((user_id, thread_id), None)
    _OWNED.discard((user_id, thread_id))
    return deleted
