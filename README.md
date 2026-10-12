# LangGraph Multi-Utility Chatbot

A Streamlit chatbot built on [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview). It keeps every conversation (SQLite), streams answers as they are generated, and can search the web, look up stock prices, do arithmetic, answer questions about a PDF you upload, and use tools from any MCP server you plug in.

## Features

- **Persistent chats**: multiple conversation threads, saved to SQLite and still there after a restart
- **Welcome guide**: an empty chat opens with a short overview of what the chatbot can do (the text lives in `welcome.md`, so you can edit it without touching code)
- **Named chats**: each chat is named from its first message (a short title written by the model) instead of showing an id
- **Streaming replies** with a live "Using tool..." status while a tool runs
- **Tools**: web search (DuckDuckGo), stock prices (Yahoo Finance via `yfinance`), calculator
- **PDF Q&A (RAG)**: upload a PDF to a chat and ask questions about it
- **Optional MCP tools**: add servers in one dictionary; a broken server is skipped, not fatal
- **Optional LangSmith tracing**

## Quick start

**Prerequisites:** Python 3.10 or newer (developed and tested on 3.13 and macOS; `.python-version` pins 3.13), [uv](https://docs.astral.sh/uv/) (recommended), and an OpenAI API key.

```bash
# 1. Create the environment and install dependencies
uv venv --python 3.13 .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt

# 2. Add your keys
cp .env.example .env                 # then open .env and fill in your values

# 3. Run the app
streamlit run streamlit_final_frontend.py
```

The app opens at <http://localhost:8501>. Without `uv`, use `python -m venv .venv`, activate it, and run `pip install -r requirements.txt`.

## Configuration

Settings live in `.env` (git-ignored). Copy [`.env.example`](.env.example) to start.

| Variable | Required | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | Yes | Chat model and PDF embeddings. Create one at platform.openai.com (API keys); the account needs credits. |
| `OPENAI_MODEL` | No | Chat model name. Defaults to `gpt-4o-mini`. |
| `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` | No | Send traces to LangSmith (commented out in `.env.example`; uncomment to enable). Each chat turn is traced as `chat_turn` with the thread id as metadata. |
| `LANGSMITH_ENDPOINT` | No | Only for non-US LangSmith accounts, e.g. `https://eu.api.smith.langchain.com`. |

No key is needed for web search or stock prices. PDF embeddings use `text-embedding-3-small` (set in the backend).

### Adding MCP tools

Edit `MCP_SERVERS` in [`langgraph_final_backend.py`](langgraph_final_backend.py), then restart the app. It is empty by default.

```python
MCP_SERVERS = {
    "math": {
        "transport": "stdio",
        "command": "python3",
        "args": ["/path/to/your/mcp-math-server/main.py"],
    },
    "remote": {
        "transport": "streamable_http",   # if this fails, try "sse"
        "url": "https://<your-server>/mcp",
    },
}
```

Each server is loaded on its own. The terminal running Streamlit prints `[MCP] Loaded N tool(s) from '<name>'` or `[MCP] Skipping server '<name>': <reason>`. Loaded tools are offered to the model automatically.

## Using the app

- Type in the chat box. Use **New Chat** to start a fresh thread; **Past conversations** in the sidebar lists your chats by name (the open one is highlighted) and reopens any of them. A new chat shows as "New chat" and opens with a welcome guide listing what the chatbot can do; the guide goes away once you send your first message, which also names the chat.
- Upload a PDF in the sidebar, then ask about it ("Summarise this document", "What does section 3 say about...?"). Each chat has its own PDF. A new upload replaces the previous one, and a scanned (image-only) PDF is rejected with a message.
- Other things to try: *"What is the price of AAPL?"*, *"Price of RELIANCE.NS"* (use the Yahoo suffix for non-US exchanges), *"What is 12.5% of 840?"*, *"Search the web for the latest LangGraph release"*.

## Project layout

```
chatbot-in-langgraph/
├── streamlit_final_frontend.py    # UI: chat, sidebar, PDF upload, streaming
├── langgraph_final_backend.py     # graph, tools, RAG, MCP loading, persistence
├── welcome.md                     # greeting shown in an empty chat (edit to change it)
├── requirements.txt               # direct dependencies
├── .env.example                   # template for your .env
├── .python-version                # Python 3.13
└── chatbot.db                     # SQLite: chat history and chat names (created on first run, git-ignored)
```

## How it works

```
 Browser
    │
    ▼
 streamlit_final_frontend.py        UI, session state, sidebar, PDF upload
    │  submit coroutine  ▲  events via queue (tokens, tool status)
    ▼                    │
 langgraph_final_backend.py         runs on its own background event loop
 ┌──────────────────────────────────────────────────────────────┐
 │  START ─► chat_node ──tool calls?──► tools ─┐                │
 │               ▲                             │                │
 │               └─────────────────────────────┘                │
 │               └──no tool calls──► END                        │
 │                                                              │
 │  chat_node: system prompt + history ─► ChatOpenAI (tools bound)
 │  tools:     web search · stock price · calculator · rag_tool · MCP tools
 │                                                              │
 │  every step is checkpointed (AsyncSqliteSaver) ─► chatbot.db │
 └──────────────────────────────────────────────────────────────┘
```

**One chat turn:** the frontend sends the user message with the chat's `thread_id` to the backend loop and reads results from a queue. `chat_node` asks the model; if the model requests tools, `ToolNode` runs them and control returns to `chat_node`, until the model answers. Tokens stream back with `stream_mode="messages"`, and a status box shows which tool ran. The checkpointer saves state under the `thread_id`, which is how history survives restarts.

**Design decisions**

- **Async graph on a background event loop.** Streamlit scripts are synchronous and rerun on every interaction, while the graph, the SQLite saver, and MCP tools are async. The backend therefore starts one long-lived event loop in a daemon thread; the frontend schedules work on it and bridges results through a `queue.Queue` into `st.write_stream`.
- **PDF Q&A is per thread.** Upload, then PyPDF extracts text, which is split (1000 characters, 200 overlap), embedded with `text-embedding-3-small`, and stored in an in-memory vector store; the top 4 chunks are retrieved per question. `rag_tool` reads the thread id from the run config rather than from a model-supplied argument, so it cannot search another chat's document.
- **Provider-neutral message handling.** The frontend reads `message.text` rather than `message.content`, which works whether a model returns a plain string or a list of content blocks.
- **Resilient MCP loading.** Servers are loaded one at a time, so one bad path or URL does not disable the rest, and the reason is printed.
- **Chat names.** When a chat gets its first message, one short extra model call (cost: a few tokens) writes a title of at most five words. It runs in the background while the reply streams and is saved in a `thread_titles` table in `chatbot.db`. If that call fails, or for chats saved before this feature existed, the first message (shortened) is used as the name.
- **Stable database location.** `chatbot.db` is anchored to the backend file's folder, so every launch location uses the same history.
- **Visible failures.** Model or network errors appear inline in the chat instead of crashing the page.
- **Tool failures don't break a chat.** A tool that raises becomes an error message the model can read and react to, web search is retried once, and a tool call left without a result (for example by a cut-off run) is closed with an error before the history goes to the model, so one bad turn can never lock a chat.

## Tech stack

| Role | Package (version tested) | Used for |
|---|---|---|
| UI | `streamlit` 1.65 | Chat UI, sidebar, file upload, streaming |
| Orchestration | `langgraph` 1.2 | `StateGraph`, `ToolNode`, conditional routing |
| Persistence | `langgraph-checkpoint-sqlite` 3.1, `aiosqlite` 0.22 | `AsyncSqliteSaver` chat history |
| LLM + embeddings | `langchain-openai` 1.7 | `ChatOpenAI`, `OpenAIEmbeddings` |
| Core types | `langchain-core` 1.6 | Messages, tools, run config, `InMemoryVectorStore` |
| PDF + splitting | `langchain-community` 0.4 (`PyPDFLoader`), `pypdf` 6, `langchain-text-splitters` 1.1 | Load and chunk PDFs |
| Web search | `langchain-community` (`DuckDuckGoSearchRun`), `ddgs` 9 | Keyless web search |
| Stock prices | `yfinance` 1.7 | Latest price from Yahoo Finance, no key |
| MCP | `langchain-mcp-adapters` 0.3 | Load tools from MCP servers |
| Config | `python-dotenv` 1.2 | Read `.env` |

## Limitations

- A PDF is indexed in memory: after the app restarts, your saved chats remain but you need to upload the PDF again. One PDF per chat.
- Chats cannot be renamed or deleted from the UI yet.
- `yfinance` is an unofficial wrapper around Yahoo Finance for personal and educational use. Yahoo can rate-limit it or change without notice, and prices can be delayed or be the last close.
- DuckDuckGo search can be rate-limited; retry later.
- `langchain-community` is archived upstream. It is still used for the DuckDuckGo tool and the PDF loader and prints a deprecation warning at startup, which is harmless.
- There is no automated test suite in this repository.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Missing credentials ... OPENAI_API_KEY` on startup | `.env` is missing or misnamed. Create it from `.env.example` in the project folder and restart. |
| `ModuleNotFoundError` / unresolved imports | Activate `.venv` and re-run `uv pip install -r requirements.txt`. In VS Code, select `.venv` as the Python interpreter. |
| Want a clean slate | Stop the app and delete `chatbot.db` (and `chatbot.db-wal` / `chatbot.db-shm` if present). |
| "Could not fetch a price for ..." | Check the ticker and exchange suffix (e.g. `RELIANCE.NS`), or retry if Yahoo is rate-limiting. |
| `[MCP] Skipping server ...` in the terminal | Check that server's `command`/`args` path or `url` in `MCP_SERVERS`. |
| "No readable text found in this PDF" | The PDF is a scan or image-only; use a version with a text layer (OCR it first). |
| Port 8501 is busy | `streamlit run streamlit_final_frontend.py --server.port 8502` |

