import os
import uuid
import tempfile

import streamlit as st
from dotenv import load_dotenv

# set_page_config must be the first Streamlit command in the script.
st.set_page_config(
    page_title="Agentic Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

load_dotenv()


def apply_streamlit_secrets():
    """Copy Streamlit Cloud secrets into environment variables.

    LangChain and the Google/Tavily SDKs read API keys from os.environ.
    Locally those keys come from `.env`. On Streamlit Community Cloud they
    come from App secrets and must be copied into the environment first.
    """
    try:
        for key, value in st.secrets.items():
            if isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    env_key = str(nested_key)
                    if not os.getenv(env_key):
                        os.environ[env_key] = str(nested_value)
                continue

            if not os.getenv(str(key)):
                os.environ[str(key)] = str(value)
    except Exception:
        # No secrets.toml locally, or Streamlit secrets are unavailable.
        pass


apply_streamlit_secrets()

from backend import (
    chatbot,
    get_all_threads,
    ingest_rag_document,
)

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage
)

from langgraph.types import Command

from ui import sidebar_nav_css


def message_to_text(content) -> str:
    """Turn Gemini / LangChain message content into a plain string.

    Newer Gemini models return a list of content blocks instead of a string:
    [{"type": "text", "text": "Hello", "index": 0}]
    Streamlit renders that list as nested JSON unless we extract the text first.
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, dict):
        block_type = content.get("type")
        if block_type in (None, "text"):
            return str(content.get("text") or content.get("content") or "")
        return ""

    if isinstance(content, list):
        return "".join(
            message_to_text(block)
            for block in content
        )

    text = getattr(content, "text", None)
    if isinstance(text, str):
        block_type = getattr(content, "type", "text")
        if block_type in (None, "text"):
            return text
        return ""

    return str(content)


USER_AVATAR = "👤"
ASSISTANT_AVATAR = "🤖"


def avatar_for(role: str) -> str:
    return USER_AVATAR if role == "user" else ASSISTANT_AVATAR


def inject_styles():
    st.markdown(
        """
        <style>
        @import url("https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap");

        html, body, [class*="css"]  {
            font-family: "DM Sans", sans-serif;
        }

        .stApp {
            background:
                radial-gradient(1200px 500px at 10% -10%, rgba(91, 140, 255, 0.18), transparent 50%),
                radial-gradient(900px 400px at 100% 0%, rgba(167, 139, 250, 0.12), transparent 45%),
                #0b1220;
        }

        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 6.5rem;
            max-width: 780px;
        }

        [data-testid="stSidebar"] {
            background: #10182a;
            border-right: 1px solid rgba(255, 255, 255, 0.06);
        }

        [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p {
            color: #c5cee0;
        }

        .app-hero {
            margin: 0 0 1.6rem 0;
        }

        .app-hero h1 {
            font-size: 2.15rem;
            line-height: 1.15;
            margin: 0 0 0.45rem 0;
            font-weight: 700;
            color: #f4f7ff;
        }

        .app-hero p {
            margin: 0;
            color: #9aa6c1;
            font-size: 1.02rem;
        }

        .empty-wrap {
            margin: 1.5rem 0 0.5rem 0;
        }

        .empty-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 16px;
            padding: 0.95rem 1rem;
            height: 100%;
        }

        .empty-card h4 {
            margin: 0 0 0.35rem 0;
            color: #e8eeff;
            font-size: 0.95rem;
        }

        .empty-card p {
            margin: 0;
            color: #8f9bb3;
            font-size: 0.86rem;
            line-height: 1.45;
        }

        [data-testid="stChatMessage"] {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 0.35rem 0.2rem;
            margin-bottom: 0.65rem;
        }

        [data-testid="stChatInput"] {
            border-top: 1px solid rgba(255, 255, 255, 0.06);
        }

        .stButton > button {
            border-radius: 12px;
            font-weight: 600;
        }

        [data-testid="stSidebar"] .stButton > button {
            justify-content: flex-start;
            text-align: left;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }
        """
        + sidebar_nav_css()
        + """
        </style>
        """,
        unsafe_allow_html=True,
    )


def short_title(text: str) -> str:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return "New chat"
    if len(cleaned) <= 42:
        return cleaned
    return cleaned[:42].rstrip() + "…"


def ensure_thread_titles():
    if "thread_titles" not in st.session_state:
        st.session_state["thread_titles"] = {}

    for thread_id in st.session_state["chat_threads"]:
        if thread_id in st.session_state["thread_titles"]:
            continue

        title = "New chat"
        for message in load_conversation(thread_id):
            if isinstance(message, HumanMessage):
                text = message_to_text(message.content).strip()
                if text:
                    title = short_title(text)
                    break

        st.session_state["thread_titles"][thread_id] = title


# Generate a unique thread ID for each new conversation
def generate_thread_id():
    return str(uuid.uuid4())


# Add a new thread ID to the conversation list
def add_thread(thread_id):

    # Prevent the same thread from being added multiple times
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


# Create a completely new chat conversation
def reset_chat():

    # Generate and assign a new thread ID
    st.session_state["thread_id"] = generate_thread_id()

    # Clear the current chat messages from the UI
    st.session_state["message_history"] = []

    # ========================= HITL ADDED =========================
    # Clear any pending human approval request
    st.session_state["pending_hitl"] = None
    # =============================================================

    # Add the new thread to the conversation list
    add_thread(st.session_state["thread_id"])
    st.session_state.setdefault("thread_titles", {})
    st.session_state["thread_titles"][st.session_state["thread_id"]] = "New chat"


# Load a previous conversation from the LangGraph checkpointer
def load_conversation(thread_id):

    # Get the saved state for the selected thread
    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    # Return saved messages
    # Return an empty list if no messages are available
    return state.values.get("messages", [])


# ========================= HITL helper functions =========================

def get_pending_interrupt(thread_id):
    """
    Return the first unresolved LangGraph interrupt for a thread.

    Returns:
        The pending Interrupt object, or None.
    """

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    try:

        # Read the current checkpoint state
        state_snapshot = chatbot.get_state(config)

        # Some LangGraph versions expose interrupts directly
        direct_interrupts = getattr(
            state_snapshot,
            "interrupts",
            ()
        ) or ()

        if direct_interrupts:
            return direct_interrupts[0]

        # Other LangGraph versions store interrupts inside tasks
        tasks = getattr(
            state_snapshot,
            "tasks",
            ()
        ) or ()

        for task in tasks:

            task_interrupts = getattr(
                task,
                "interrupts",
                ()
            ) or ()

            if task_interrupts:
                return task_interrupts[0]

    except Exception:

        # A newly created thread may not have a checkpoint yet
        return None

    return None


def save_pending_interrupt(thread_id, interrupt_object):
    """
    Save the pending interrupt information inside Streamlit state.
    """

    st.session_state["pending_hitl"] = {
        "thread_id": thread_id,
        "prompt": str(interrupt_object.value)
    }


def sync_pending_interrupt(thread_id):
    """
    Synchronize Streamlit HITL state with the LangGraph checkpoint.

    This allows a pending approval request to reappear after:
    - a Streamlit rerun
    - a browser refresh
    - switching between conversations
    """

    pending_interrupt = get_pending_interrupt(thread_id)

    if pending_interrupt is not None:

        save_pending_interrupt(
            thread_id,
            pending_interrupt
        )

    else:

        current_pending = st.session_state.get(
            "pending_hitl"
        )

        if (
            current_pending is not None
            and current_pending.get("thread_id") == thread_id
        ):
            st.session_state["pending_hitl"] = None


def resume_hitl_execution(decision):
    """
    Resume an interrupted LangGraph execution.

    Args:
        decision:
            "yes" approves the stock purchase.
            "no" rejects the stock purchase.
    """

    pending_hitl = st.session_state.get(
        "pending_hitl"
    )

    if not pending_hitl:

        st.warning(
            "There is no pending action to approve or reject."
        )

        return

    # Get the thread that originally triggered the interrupt
    interrupted_thread_id = pending_hitl["thread_id"]

    # The same thread ID must be used when resuming
    resume_config = {
        "configurable": {
            "thread_id": interrupted_thread_id
        },
        "metadata": {
            "thread_id": interrupted_thread_id
        },
        "run_name": "hitl_resume_trace",
    }

    try:

        # Display the resumed response
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):

            status_holder = {
                "box": st.status(
                    "🔄 Resuming the requested action...",
                    expanded=True
                )
            }

            def resumed_ai_only_stream():

                # Resume the graph with the human decision
                for message_chunk, metadata in chatbot.stream(
                    Command(resume=decision),
                    config=resume_config,
                    stream_mode="messages",
                ):

                    # Update tool execution status
                    if isinstance(
                        message_chunk,
                        ToolMessage
                    ):

                        tool_name = getattr(
                            message_chunk,
                            "name",
                            "tool"
                        )

                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                    # Stream only assistant-generated text
                    if isinstance(
                        message_chunk,
                        AIMessage
                    ):
                        text = message_to_text(message_chunk.content)
                        if text:
                            yield text

            # Display the streamed final answer
            resumed_ai_message = st.write_stream(
                resumed_ai_only_stream()
            )

            # Check whether another interrupt occurred
            next_interrupt = get_pending_interrupt(
                interrupted_thread_id
            )

            if next_interrupt is not None:

                save_pending_interrupt(
                    interrupted_thread_id,
                    next_interrupt
                )

                status_holder["box"].update(
                    label="⚠️ Another approval is required",
                    state="complete",
                    expanded=False
                )

            else:

                # No more pending approval
                st.session_state["pending_hitl"] = None

                status_holder["box"].update(
                    label="✅ Action completed",
                    state="complete",
                    expanded=False
                )

        # Save the assistant response in Streamlit UI history
        if resumed_ai_message:

            st.session_state["message_history"].append({
                "role": "assistant",
                "content": message_to_text(resumed_ai_message)
            })

        # Rerun so the response appears in normal chat order
        st.rerun()

    except Exception as error:

        st.error(
            f"Could not resume the requested action: {error}"
        )


# ========================= Page configuration =========================

inject_styles()

st.markdown(
    """
    <div class="app-hero">
        <h1>Agentic Chatbot</h1>
        <p>Ask questions, attach a PDF, or request a stock purchase that waits for your approval.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# Create message_history when the app runs for the first time
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []


# Create a thread ID when the app runs for the first time
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()


# Create a list for storing all conversation thread IDs
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()


# ========================= HITL ADDED =========================

# Store the currently pending human approval request
if "pending_hitl" not in st.session_state:
    st.session_state["pending_hitl"] = None

# =============================================================


# Add the current thread to the conversation list
add_thread(st.session_state["thread_id"])
ensure_thread_titles()


# ========================= HITL ADDED =========================

# Recover pending approval after page refresh or rerun
sync_pending_interrupt(
    st.session_state["thread_id"]
)

# =============================================================


# ========================= Sidebar threading feature =========================

with st.sidebar:
    st.markdown("### Conversations")
    st.caption("Each thread keeps its own memory in SQLite.")

    if st.button("＋  New chat", type="primary", use_container_width=True):
        reset_chat()
        st.rerun()

    st.divider()

    for thread_id in st.session_state["chat_threads"][::-1]:
        label = st.session_state.get("thread_titles", {}).get(
            thread_id,
            f"Chat {thread_id[:8]}"
        )
        is_active = thread_id == st.session_state["thread_id"]

        if st.button(
            label,
            key=thread_id,
            type="primary" if is_active else "secondary",
            use_container_width=True,
        ):
            st.session_state["thread_id"] = thread_id
            messages = load_conversation(thread_id)
            temp_messages = []

            for message in messages:
                if isinstance(message, HumanMessage):
                    role = "user"
                elif isinstance(message, AIMessage):
                    role = "assistant"
                else:
                    continue

                content = message_to_text(message.content)
                if not content:
                    continue

                temp_messages.append({
                    "role": role,
                    "content": content
                })

            st.session_state["message_history"] = temp_messages
            sync_pending_interrupt(thread_id)
            st.rerun()


# ========================= Main chat interface =========================

if not st.session_state["message_history"]:
    st.markdown('<div class="empty-wrap">', unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(
            """
            <div class="empty-card">
                <h4>Ask anything</h4>
                <p>Weather, stocks, math, or a live web search.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            """
            <div class="empty-card">
                <h4>Attach a PDF</h4>
                <p>Index a document, then ask questions about it.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_c:
        st.markdown(
            """
            <div class="empty-card">
                <h4>Approve trades</h4>
                <p>Stock purchases pause until you confirm.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"], avatar=avatar_for(message["role"])):
        st.markdown(message_to_text(message["content"]))


# ========================= HITL approval interface =========================

# Get the currently pending approval request
pending_hitl = st.session_state.get(
    "pending_hitl"
)

# Check whether the pending approval belongs to
# the currently selected conversation
current_thread_has_pending_hitl = (
    pending_hitl is not None
    and pending_hitl.get("thread_id")
    == st.session_state["thread_id"]
)


if current_thread_has_pending_hitl:
    with st.container(border=True):
        st.markdown("#### Approval needed")
        st.write(pending_hitl["prompt"])
        approve_column, reject_column = st.columns(2)
        with approve_column:
            if st.button(
                "Approve purchase",
                key=f"approve_{st.session_state['thread_id']}",
                type="primary",
                use_container_width=True
            ):
                resume_hitl_execution("yes")
        with reject_column:
            if st.button(
                "Reject purchase",
                key=f"reject_{st.session_state['thread_id']}",
                use_container_width=True
            ):
                resume_hitl_execution("no")


# ========================= Fixed chat input with PDF upload =========================

# Keep st.chat_input directly in the main body.
# This keeps it fixed at the bottom of the screen.
#
# accept_file=True adds the attachment button inside the chat input.
# file_type=["pdf"] allows PDF files only.
submission = st.chat_input(
    "Ask a question or attach a PDF…",
    accept_file=True,
    file_type=["pdf"],
    disabled=current_thread_has_pending_hitl
)


# Default user input value
user_input = None


# Process the submitted text and PDF
if submission:

    # Get the text entered by the user
    user_input = submission.text

    # Get the uploaded files
    # This is always a list when accept_file is enabled
    uploaded_files = submission.files

    # Process the uploaded PDF if one was attached
    if uploaded_files:

        uploaded_pdf = uploaded_files[0]

        # Store the temporary file path
        temporary_file_path = None

        try:

            # Save the uploaded PDF as a temporary local file
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temporary_file:

                temporary_file.write(
                    uploaded_pdf.getvalue()
                )

                temporary_file_path = temporary_file.name

            # Call the existing backend RAG ingestion function
            with st.spinner(
                f"Processing {uploaded_pdf.name}..."
            ):

                ingest_rag_document(
                    temporary_file_path
                )

            # Display PDF processing confirmation
            st.toast(
                f"{uploaded_pdf.name} processed successfully.",
                icon="✅"
            )

        except Exception as error:

            # Display PDF processing error
            st.error(
                f"PDF processing failed: {error}"
            )

        finally:

            # Delete the temporary PDF after indexing
            if (
                temporary_file_path
                and os.path.exists(temporary_file_path)
            ):
                os.remove(temporary_file_path)


# Run this block after the user submits a text message
if user_input:
    tid = st.session_state["thread_id"]
    titles = st.session_state.setdefault("thread_titles", {})
    if titles.get(tid) in (None, "New chat", f"Chat {tid[:8]}"):
        titles[tid] = short_title(user_input)

    # Save the user's message in Streamlit session state
    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    })

    # Display the user's message in the chat interface
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_input)

    # Pass the current thread ID to LangGraph
    # LangGraph uses this ID to save and retrieve conversation memory
    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        },
        "metadata": {
            "thread_id": st.session_state["thread_id"]
        },
        "run_name": "chat_trace",
    }

    # Assistant streaming block
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):

        # Use a mutable holder so the generator can set/modify it
        status_holder = {
            "box": None
        }

        def ai_only_stream():

            for message_chunk, metadata in chatbot.stream(
                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },
                config=CONFIG,
                stream_mode="messages",
            ):

                # Lazily create & update the SAME status container
                # when any tool runs
                if isinstance(
                    message_chunk,
                    ToolMessage
                ):

                    tool_name = getattr(
                        message_chunk,
                        "name",
                        "tool"
                    )

                    if status_holder["box"] is None:

                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …",
                            expanded=True
                        )

                    else:

                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream ONLY assistant tokens
                if isinstance(
                    message_chunk,
                    AIMessage
                ):
                    text = message_to_text(message_chunk.content)
                    if text:
                        yield text

            # ========================= HITL ADDED =========================

            # interrupt() pauses the graph without returning
            # a completed ToolMessage.
            #
            # Inspect the saved checkpoint after streaming ends.
            pending_interrupt = get_pending_interrupt(
                st.session_state["thread_id"]
            )

            if pending_interrupt is not None:

                # Save the interrupt for displaying approval buttons
                save_pending_interrupt(
                    st.session_state["thread_id"],
                    pending_interrupt
                )

                yield (
                    "\n\n⚠️ This stock purchase requires your approval. "
                    "Use the Approve Purchase or Reject Purchase "
                    "button below."
                )

            # =============================================================

        ai_message = st.write_stream(
            ai_only_stream()
        )

        # Finalize only if a tool was actually used
        if status_holder["box"] is not None:

            # Check whether execution is waiting for approval
            if get_pending_interrupt(
                st.session_state["thread_id"]
            ) is not None:

                status_holder["box"].update(
                    label="⏸️ Waiting for human approval",
                    state="complete",
                    expanded=False
                )

            else:

                status_holder["box"].update(
                    label="✅ Tool finished",
                    state="complete",
                    expanded=False
                )

    # Save the complete assistant response in Streamlit session state
    st.session_state["message_history"].append({
        "role": "assistant",
        "content": message_to_text(ai_message)
    })

    # ========================= HITL ADDED =========================

    # Approval controls are rendered earlier in the script.
    # Rerun so they appear immediately after interrupt().
    if (
        st.session_state.get("pending_hitl") is not None
        and st.session_state["pending_hitl"].get("thread_id")
        == st.session_state["thread_id"]
    ):
        st.rerun()

    # =============================================================