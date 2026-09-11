import streamlit as st

from pathlib import Path

from ui import sidebar_nav_css

st.set_page_config(
    page_title="Implementation",
    page_icon="🤖",
    layout="centered",
)

st.markdown(
    f"""
    <style>
    .block-container {{ padding-top: 1.6rem; padding-bottom: 4rem; max-width: 860px; }}
    header[data-testid="stHeader"] {{ background: transparent; }}
    h2 {{ margin-top: 1.8rem; }}
    [data-testid="stSidebar"] {{ background: #10182a; }}
    .flow-wrap svg {{ width: 100%; height: auto; display: block; margin: 0.5rem 0 1.25rem 0; }}
    {sidebar_nav_css()}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Implementation")
st.caption("How this chatbot is built, what it does, and which tools sit behind each step.")

st.markdown(
    """
This is a **learning project**: a Streamlit chat UI in front of a LangGraph agent.
The agent can answer directly, or call tools for search, weather, stocks, math, PDFs, and
human-approved purchases. Chat memory is SQLite. PDF answers use FAISS.
"""
)

st.subheader("What is implemented")

col1, col2 = st.columns(2)
with col1:
    st.markdown(
        """
- Multi-turn chat with streaming replies
- Conversation threads in the sidebar
- Tool-using agent (LangGraph)
- Switchable LLM: Gemini or OpenAI
- Web search, weather, stock quotes, calculator
        """
    )
with col2:
    st.markdown(
        """
- PDF upload and RAG answers
- Human-in-the-loop stock purchase
- SQLite checkpoint memory
- Local run and Streamlit Community Cloud
- Optional LangSmith tracing
        """
    )

st.subheader("How a message flows")

st.markdown(
    """
1. The user types in **Streamlit** (`streamlit_app.py`), optionally attaching a PDF.
2. A PDF is indexed into **FAISS** immediately. It never goes through the graph.
3. Text is sent to **LangGraph** with a `thread_id`.
4. The graph loads that thread from **SQLite**.
5. `chat_node` calls the LLM. The model either answers, or requests a tool.
6. If a tool is needed, the `tools` node runs it (Tavily, OpenWeather, and so on).
7. Control returns to `chat_node`. The LLM writes the user-facing answer from the tool result.
8. The thread is saved back to SQLite. Streamlit shows the reply.
    """
)

st.markdown(
    '<div class="flow-wrap">'
    + Path("docs/message_flow.svg").read_text(encoding="utf-8")
    + "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    """
The LLM is the **same** `chat_node` both times. First call: decide whether to use a tool.
`ToolNode` runs the tool and does **not** call the model. Second call: turn the tool
result into a reply. If no tool is needed, the model runs only once.
"""
)

st.subheader("LangGraph")

st.markdown(
    """
The graph in `backend.py` is small on purpose:

`START → chat_node ⇄ tools → END`

`tools_condition` looks at the last AI message. Tool calls go to `tools`. Otherwise the
graph ends. After tools run, an edge always returns to `chat_node`.
"""
)

st.subheader("Tools")

st.markdown(
    """
| Tool | How it is implemented |
| --- | --- |
| `search_tool` | Tavily Search API |
| `calculator` | Local `eval` with a small allow-list (`math`, `abs`, …) |
| `get_current_weather` | OpenWeather geocoding + current weather |
| `get_stock_price` | Alpha Vantage global quote |
| `rag_tool` | Embed the question, retrieve 4 FAISS chunks, return them to the LLM |
| `purchase_stock` | Simulated. `interrupt()` pauses until Streamlit sends yes/no |
"""
)

st.subheader("PDF / RAG")

st.markdown(
    """
Upload and question answering are separate.

1. **Upload** — `ingest_rag_document` splits the PDF, embeds chunks, and writes FAISS.
2. **Question** — the LLM only sees the user's text, plus a system prompt that says
   to use `rag_tool` for document questions.
3. **Retrieval** — `rag_tool` returns the closest chunks. The next `chat_node` call
   answers from those snippets.

The model is not given the whole PDF. A new upload replaces the previous FAISS index.
There is one shared index, not one per conversation.
"""
)

st.subheader("Memory")

st.markdown(
    """
LangGraph `SqliteSaver` writes checkpoints to `chatbot.db`. Each sidebar chat is a
`thread_id`. Switching chats calls `get_state` for that thread. Restarting the app
keeps history as long as the SQLite file is still there.
"""
)

st.subheader("Human-in-the-loop")

st.markdown(
    """
`purchase_stock` calls `interrupt(...)`. The graph stops and Streamlit shows Approve /
Reject. Chat input is disabled until then. A click resumes with
`Command(resume="yes")` or `"no"`. The tool returns success or cancelled, then
`chat_node` runs again.
"""
)

st.subheader("Technologies")

st.markdown(
    """
| Layer | Technology |
| --- | --- |
| UI | Streamlit |
| Agent | LangGraph + LangChain |
| LLM | Google Gemini (`gemini-3.6-flash`) or OpenAI (`gpt-4o-mini` by default) |
| Chat memory | SQLite via `langgraph-checkpoint-sqlite` |
| PDF retrieval | pypdf, recursive splitter, FAISS, Gemini or OpenAI embeddings |
| Search / weather / stocks | Tavily, OpenWeather, Alpha Vantage |
| Config | `.env` locally, Streamlit secrets on Cloud |
| Tracing | LangSmith (optional) |
"""
)

st.subheader("Project files")

st.markdown(
    """
| File | Role |
| --- | --- |
| `streamlit_app.py` | Chat UI, threads, PDF upload, approval buttons |
| `backend.py` | Graph, tools, LLM, SQLite, FAISS |
| `pages/1_Implementation.py` | This page |
| `.env` | Local API keys (`LLM_PROVIDER` chooses Gemini or OpenAI) |
| `chatbot.db` | Conversation checkpoints |
| `faiss_db_*` | Embedded PDF chunks |
"""
)

st.subheader("Run it")

st.code(
    "python3.11 -m venv .venv\n"
    "source .venv/bin/activate\n"
    "pip install -r requirements.txt\n"
    "cp .env.example .env   # add API keys\n"
    "streamlit run streamlit_app.py",
    language="bash",
)
