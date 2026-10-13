# LangGraph Chatbot: Postgres + pgvector version

The same chatbot as the SQLite version in the parent folder (chat, web search, stock prices, calculator, PDF Q&A, named and deletable chats, welcome guide), rebuilt so that **everything is stored in one Postgres database** and **every row belongs to a user**.

It lives in its own folder with its own frontend, backend, requirements and Docker setup, so you can run it and the SQLite version separately and compare them. The SQLite version is untouched.

## Run it

You need Docker (for a local database), [uv](https://docs.astral.sh/uv/), and an OpenAI API key.

```bash
cd postgres_version

# 1. Start a local Postgres with the pgvector extension (port 5433)
docker compose up -d

# 2. Create this folder's own environment and install its dependencies
uv venv --python 3.13 .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt

# 3. Add your keys
cp .env.example .env                 # then put your OPENAI_API_KEY in .env

# 4. Run the app with THIS folder's environment
.venv/bin/streamlit run streamlit_pg_frontend.py     # Windows: .venv\Scripts\streamlit run streamlit_pg_frontend.py
```

Use the explicit `.venv/bin/streamlit` path in step 4. It guarantees this folder's environment even if your terminal has another one active (VS Code, for example, may activate the main project's `.venv` for you, which does not have the Postgres packages).

To run both versions at the same time, give one a different port, e.g. `.venv/bin/streamlit run streamlit_pg_frontend.py --server.port 8502`. They use different storage (a SQLite file vs. Postgres), so their chats are separate.

## Troubleshooting

| What you see | Cause | Fix |
|---|---|---|
| `Could not start the backend: No module named 'psycopg'` | Streamlit is running from a different Python environment (often the main project's `.venv`) | From this folder run `.venv/bin/streamlit run streamlit_pg_frontend.py`. If there is no `.venv` here yet, do step 2 above. |
| `Could not start the backend: ... Connection refused` (port 5433) | The database is not running | `docker compose up -d` in this folder, then reload the page. |
| `DATABASE_URL is not set` | There is no `.env` in this folder | `cp .env.example .env` and fill it in. |

Stop the database with `docker compose down` (add `-v` to also delete its data). To use a hosted Postgres instead of Docker, put its connection string in `DATABASE_URL`; it needs the `vector` extension.

## How it differs from the SQLite version

| | SQLite version (parent folder) | Postgres version (this folder) |
|---|---|---|
| Storage | one `chatbot.db` file | one Postgres database |
| Chat history | `AsyncSqliteSaver` | `AsyncPostgresSaver` (wrapped so it checks ownership) |
| Chat list and names | found by scanning every saved checkpoint, plus a titles table | a `chats` table, indexed per user |
| PDF vectors | text and embeddings saved as bytes; an in-memory index is rebuilt on use | a `vector(1536)` column, searched inside the database with pgvector |
| Users | none: everyone sees every chat | every row has a `user_id`, every query is filtered by it, and the storage layer refuses other users |
| Concurrency | one file, one writer | a connection pool for many users |
| Setup | one key | one key plus a database (`docker compose up -d`) |
| Hosting | needs a persistent disk | any Postgres that has pgvector |

## How the data is stored

| Table | What it holds |
|---|---|
| `chats` | one row per chat: `thread_id`, `user_id`, `title`, timestamps. The chat list is a query on this table. |
| `pdf_documents` | one row per chat that has a PDF: file name, page and chunk counts, the embedding model used |
| `pdf_chunks` | one row per PDF chunk: its text, metadata and a `vector(1536)` embedding |
| `checkpoints`, `checkpoint_blobs`, `checkpoint_writes` | LangGraph's own tables, holding each chat's messages |

Deleting a `chats` row also deletes that chat's PDF rows (`ON DELETE CASCADE`).

PDF search is one query: `ORDER BY embedding <=> query_vector LIMIT 4`, where `<=>` is pgvector's cosine distance, filtered to one user's one chat. There is deliberately **no vector index** (such as HNSW): each search is first narrowed to a single chat's roughly 100 chunks, so an exact scan is instant. An index only pays off when searching across very large collections.

## How users are kept apart

- Every query filters on `user_id`, and the graph's config must carry both a `user_id` and a `thread_id`, otherwise nothing runs.
- Ownership is enforced **in the storage layer** (`OwnedPostgresSaver`): any read or write of a chat's history first checks the chat belongs to the caller. This matters because LangGraph saves a run's input before any node runs, so a check inside a node would be too late.
- Listing, opening, naming, uploading a PDF, searching a PDF and deleting all refuse another user's chat.

**Important: there is no login yet.** The app currently runs as one fixed user (`APP_USER_ID`, default `local-user`), so anyone who opens the page is that user. The data layer is ready for many users, but do not put this online for several people until a login assigns each person their own `user_id`.

## Configuration (`.env` in this folder)

| Variable | Required | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | Yes | Chat model and PDF embeddings |
| `DATABASE_URL` | Yes | The Postgres connection string (the default in `.env.example` matches `docker-compose.yml`) |
| `OPENAI_MODEL` | No | Chat model, default `gpt-4o-mini` |
| `APP_USER_ID` | No | Who the app runs as until a login exists, default `local-user` |
| `LANGSMITH_*` | No | Tracing, as in the parent README |

The development database password in `docker-compose.yml` is `chatbot`; use a strong one for anything shared or online.

## Notes and limitations

- Chat names, PDF details and ownership checks are cached per process. That is fine for one app process; with several copies running, a change made by one may reach another a little late.
- There is no migration from the SQLite version's chats yet; this version starts empty.
- One PDF per chat (a new upload replaces it). A PDF's text is stored in the database.
- Tested against Postgres 17 with pgvector in Docker. Hosted providers have not been tried.
- There is no automated test suite in this repository.

## Files

```
postgres_version/
├── streamlit_pg_frontend.py   # the UI (same look as the SQLite version, but user-aware)
├── langgraph_pg_backend.py    # graph, tools, Postgres storage, pgvector search, ownership checks
├── welcome.md                 # the greeting shown in an empty chat
├── requirements.txt           # this version's own dependencies
├── docker-compose.yml         # local Postgres + pgvector
└── .env.example               # template for .env
```
