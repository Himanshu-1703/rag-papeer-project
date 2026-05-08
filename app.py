import json
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from backend.rag_graph import build_graph

st.set_page_config(page_title="Papeer", page_icon="📚", layout="centered")


@st.cache_resource
def get_graph():
    return build_graph()


SESSIONS_FILE = Path("sessions.json")
_rename_llm = ChatOpenAI(model="gpt-5-mini")


def load_sessions() -> dict:
    try:
        return json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_sessions(sessions_meta: dict) -> None:
    SESSIONS_FILE.write_text(json.dumps(sessions_meta, indent=2), encoding="utf-8")


def _serialize_state(values: dict) -> dict:
    out = {}
    for k, v in values.items():
        if k == "messages":
            out[k] = [
                {
                    "type": type(m).__name__,
                    "content": (
                        m.content[:300]
                        if isinstance(m.content, str)
                        else repr(m.content)[:300]
                    ),
                }
                for m in (v or [])
            ]
        elif k == "retrieved_docs":
            out[k] = [
                {"content": d.page_content[:300], "metadata": d.metadata}
                for d in (v or [])
            ]
        else:
            out[k] = v
    return out


def generate_session_name(first_message: str) -> str:
    try:
        response = _rename_llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "Generate a concise 3-5 word title for a research chat session "
                        "based on the user's first message. Return only the title, "
                        "no punctuation at the end, no quotes."
                    ),
                },
                {"role": "user", "content": first_message[:500]},
            ]
        )
        return response.content.strip()
    except Exception:
        return "New Session"


def maybe_rename_session(session_id: str, first_message: str) -> None:
    if st.session_state.sessions_meta.get(session_id, {}).get("is_named"):
        return
    name = generate_session_name(first_message)
    st.session_state.sessions_meta[session_id]["name"] = name
    st.session_state.sessions_meta[session_id]["is_named"] = True
    save_sessions(st.session_state.sessions_meta)


def create_session() -> str:
    sid = str(uuid.uuid4())
    st.session_state.sessions_meta[sid] = {
        "id": sid,
        "name": "New Session",
        "created_at": datetime.utcnow().isoformat(),
        "is_named": False,
    }
    save_sessions(st.session_state.sessions_meta)
    st.session_state.chats[sid] = []
    st.session_state.turns[sid] = 0
    return sid


def switch_session(session_id: str) -> None:
    st.session_state.active_session_id = session_id
    if session_id not in st.session_state.chats:
        st.session_state.chats[session_id] = []
    if session_id not in st.session_state.turns:
        st.session_state.turns[session_id] = 0


graph = get_graph()

# ── Bootstrap ──────────────────────────────────────────────────────────────────
if "sessions_meta" not in st.session_state:
    st.session_state.sessions_meta = load_sessions()
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "turns" not in st.session_state:
    st.session_state.turns = {}
if "active_session_id" not in st.session_state:
    if st.session_state.sessions_meta:
        latest = max(
            st.session_state.sessions_meta.values(),
            key=lambda s: s["created_at"],
        )
        switch_session(latest["id"])
    else:
        sid = create_session()
        st.session_state.active_session_id = sid

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    if st.button("+ New Chat", use_container_width=True):
        new_sid = create_session()
        st.session_state.active_session_id = new_sid
        st.rerun()
    st.divider()
    
    st.markdown("## 💬 Sessions")

    sorted_sessions = sorted(
        st.session_state.sessions_meta.values(),
        key=lambda s: s["created_at"],
        reverse=True,
    )
    for session in sorted_sessions:
        sid = session["id"]
        is_active = sid == st.session_state.active_session_id
        btn_type = "primary" if is_active else "secondary"
        if st.button(
            session["name"],
            key=f"sess_{sid}",
            use_container_width=True,
            type=btn_type,
        ):
            if not is_active:
                switch_session(sid)
                st.rerun()

# ── Page header ────────────────────────────────────────────────────────────────
st.title("📚 Papeer — Research Paper Assistant")
st.markdown(
    "🔍 **Ask questions** from your uploaded papers &nbsp;·&nbsp; "
    "✅ **Verify claims** against recent literature &nbsp;·&nbsp; "
    "🌐 **Search the web** for the latest findings\n\n"
    "> Upload papers in the sidebar *(coming soon)* and start chatting below."
)
st.divider()

# ── Chat display ───────────────────────────────────────────────────────────────
active_sid = st.session_state.active_session_id
for msg in st.session_state.chats.get(active_sid, []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            with st.expander(f"📊 Graph state · turn {msg['turn']}", expanded=False):
                st.json(msg["graph_state"])

# ── Chat input ─────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about your papers, verify a claim, or search the web…"):
    active_sid = st.session_state.active_session_id
    if active_sid not in st.session_state.chats:
        st.session_state.chats[active_sid] = []
    if active_sid not in st.session_state.turns:
        st.session_state.turns[active_sid] = 0

    is_first_message = len(st.session_state.chats[active_sid]) == 0

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chats[active_sid].append({"role": "user", "content": prompt})
    st.session_state.turns[active_sid] += 1
    current_turn = st.session_state.turns[active_sid]

    if is_first_message:
        maybe_rename_session(active_sid, prompt)

    input_state = {
        "messages": [HumanMessage(content=prompt)],
        "session_id": active_sid,
        "query": prompt,
        "route": None,
        "retrieved_docs": [],
        "retrieval_attempts": 0,
        "claim_verdict": None,
        "claim_source": None,
        "superseding_papers": [],
        "answer": None,
        "is_relevant": None,
        "rewrite_count": 0,
    }
    config = {"configurable": {"thread_id": active_sid}}

    with st.chat_message("assistant"):
        placeholder = st.empty()
        response_text = ""

        for chunk, metadata in graph.stream(input_state, config, stream_mode="messages"):
            if (
                metadata.get("langgraph_node") == "generate_answer"
                and hasattr(chunk, "content")
                and chunk.content
            ):
                response_text += chunk.content
                placeholder.markdown(response_text + "▌")

        if not response_text:
            final_values = graph.get_state(config).values
            response_text = final_values.get("answer") or "No response generated."

        placeholder.markdown(response_text)

        final_values = graph.get_state(config).values
        state_snapshot = _serialize_state(final_values)

        with st.expander(f"📊 Graph state · turn {current_turn}", expanded=False):
            st.json(state_snapshot)

    st.session_state.chats[active_sid].append(
        {
            "role": "assistant",
            "content": response_text,
            "graph_state": state_snapshot,
            "turn": current_turn,
        }
    )

    if is_first_message:
        st.rerun()
