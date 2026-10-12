import queue
import uuid
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langgraph_final_backend import (
    chatbot,
    delete_chat,
    get_thread_title,
    ingest_pdf,
    load_messages,
    mcp_tools,
    request_title,
    retrieve_all_threads,
    submit_async_task,
    thread_document_metadata,
)


# =========================== Utilities ===========================
def generate_thread_id():
    return str(uuid.uuid4())


def thread_label(thread_id):
    # A chat has no title until its first message is sent. "$" would start a LaTeX
    # expression in a button label, so it is escaped.
    return (get_thread_title(thread_id) or "New chat").replace("$", "\\$")


WELCOME_FILE = Path(__file__).with_name("welcome.md")


def show_welcome():
    """The greeting shown in an empty chat. Its text lives in welcome.md."""
    try:
        text = WELCOME_FILE.read_text(encoding="utf-8")
    except OSError:
        return  # no welcome file: just skip the greeting
    with st.container(border=True):
        st.markdown(text)
        if mcp_tools:
            st.caption("Extra tools connected via MCP: " + ", ".join(t.name for t in mcp_tools))


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].insert(0, thread_id)  # newest first


def drop_current_if_unused():
    """Forget the current chat if nothing was ever sent in it (it has no title yet)."""
    current = st.session_state["thread_id"]
    if get_thread_title(current) is None and current in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].remove(current)


def start_new_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


def reset_chat():
    if get_thread_title(st.session_state["thread_id"]) is None:
        return  # already in an empty chat
    start_new_chat()


def open_thread(thread_id):
    drop_current_if_unused()
    st.session_state["thread_id"] = thread_id
    history = []
    for msg in load_messages(thread_id):
        if not isinstance(msg, (HumanMessage, AIMessage)) or not msg.text:
            continue  # skip tool results and tool-call turns that carry no text
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        history.append({"role": role, "content": msg.text})
    st.session_state["message_history"] = history


def remove_chat(thread_id):
    delete_chat(thread_id)
    chats = st.session_state["chat_threads"]
    if thread_id in chats:
        chats.remove(thread_id)
    if thread_id == st.session_state["thread_id"]:
        start_new_chat()  # the open chat was deleted: carry on in a fresh one
    st.toast("Chat deleted")


def stream_reply(user_input, config, status_holder):
    """Run one chat turn on the backend event loop and yield the reply text as it streams."""
    events = queue.Queue()

    async def run_stream():
        try:
            async for message_chunk, _ in chatbot.astream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages",
            ):
                events.put(("message", message_chunk))
        except Exception as exc:
            events.put(("error", exc))
        finally:
            events.put(("done", None))

    submit_async_task(run_stream())

    while True:
        kind, payload = events.get()
        if kind == "done":
            break
        if kind == "error":
            raise payload

        # Lazily create & update the SAME status container when any tool runs
        if isinstance(payload, ToolMessage):
            tool_name = getattr(payload, "name", "tool")
            if status_holder["box"] is None:
                status_holder["box"] = st.status(f"🔧 Using `{tool_name}` …", expanded=True)
            else:
                status_holder["box"].update(
                    label=f"🔧 Using `{tool_name}` …", state="running", expanded=True
                )

        # Stream ONLY assistant tokens
        if isinstance(payload, AIMessage):
            yield payload.text


# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "failed_uploads" not in st.session_state:
    st.session_state["failed_uploads"] = {}

add_thread(st.session_state["thread_id"])

thread_key = st.session_state["thread_id"]

# ============================ Sidebar ============================
st.sidebar.title("LangGraph Chatbot")
current_chat = st.sidebar.empty()  # filled in at the end, once the chat has a title

st.sidebar.button("New Chat", width="stretch", on_click=reset_chat)

pdf_status = st.sidebar.empty()  # filled in below, once any upload has been processed

# One uploader per thread, so switching chats doesn't re-index the same file.
uploaded_pdf = st.sidebar.file_uploader(
    "Upload a PDF for this chat", type=["pdf"], key=f"pdf-{thread_key}"
)
failed_uploads = st.session_state["failed_uploads"]
if uploaded_pdf:
    if uploaded_pdf.name == thread_document_metadata(thread_key).get("filename"):
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    elif uploaded_pdf.file_id not in failed_uploads:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            try:
                ingest_pdf(
                    uploaded_pdf.getvalue(),
                    thread_id=thread_key,
                    filename=uploaded_pdf.name,
                )
            except Exception as exc:
                failed_uploads[uploaded_pdf.file_id] = (
                    f"Could not index `{uploaded_pdf.name}`: {exc}"
                )
                status_box.update(label="❌ Indexing failed", state="error", expanded=False)
            else:
                status_box.update(label="✅ PDF indexed", state="complete", expanded=False)
    if uploaded_pdf.file_id in failed_uploads:
        st.sidebar.error(failed_uploads[uploaded_pdf.file_id])

doc_meta = thread_document_metadata(thread_key)
if doc_meta:
    pdf_status.success(
        f"Using `{doc_meta.get('filename')}` "
        f"({doc_meta.get('chunks')} chunks from {doc_meta.get('documents')} pages)"
    )
else:
    pdf_status.info("No PDF indexed yet.")

st.sidebar.subheader("Past conversations")
threads_box = st.sidebar.container()  # filled in at the end, so a new chat's name shows at once

# ============================ Main Layout ========================
st.title("Multi Utility Chatbot")

user_input = st.chat_input("Ask about your document or use tools")

# Greeting in an empty chat; it goes away with the first message
if not st.session_state["message_history"] and not user_input:
    show_welcome()

# Render history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

title_future = None

if user_input:
    # The first message of a chat names it. The title is written while the reply streams.
    if get_thread_title(thread_key) is None:
        title_future = request_title(thread_key, user_input)

    # Show user's message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    # Assistant streaming block
    with st.chat_message("assistant"):
        # Use a mutable holder so the generator can set/modify it
        status_holder = {"box": None}

        try:
            ai_message = st.write_stream(stream_reply(user_input, CONFIG, status_holder))
        except Exception as exc:
            st.error(f"Something went wrong: {exc}")
        else:
            # Finalize only if a tool was actually used
            if status_holder["box"] is not None:
                status_holder["box"].update(
                    label="✅ Tool finished", state="complete", expanded=False
                )
            if ai_message:
                st.session_state["message_history"].append(
                    {"role": "assistant", "content": ai_message}
                )

# ======================= Chat names (rendered last) ===============
if title_future is not None:
    try:
        title_future.result(timeout=3)  # normally done by the time the reply has streamed
    except Exception:
        pass  # the first message stands in as the title; the model's title shows next time

current_chat.markdown(f"**Chat:** {thread_label(thread_key)}")

with threads_box:
    for thread_id in st.session_state["chat_threads"]:
        chat_col, delete_col = st.columns([5, 1], vertical_alignment="center")
        chat_col.button(
            thread_label(thread_id),
            key=f"thread-{thread_id}",
            on_click=open_thread,
            args=(thread_id,),
            type="primary" if thread_id == thread_key else "secondary",
            width="stretch",
        )
        if get_thread_title(thread_id) is not None:  # an empty new chat has nothing to delete
            with delete_col.popover("🗑️", key=f"delete-menu-{thread_id}", width="stretch"):
                st.write("Delete this chat? Its messages and PDF are removed for good.")
                st.button(
                    "Delete",
                    key=f"delete-{thread_id}",
                    type="primary",
                    on_click=remove_chat,
                    args=(thread_id,),
                )
