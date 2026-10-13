import itertools
import json
import os
import time
import uuid
from datetime import datetime

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")
CHATS_FILE = "chats.json"

st.set_page_config(
    page_title="TN-AGRI BOT | Tamil Nadu Agriculture Schemes",
    page_icon="🌾",
    layout="centered",
)

# ---------- Custom Styles ----------
st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 50%, #558b2f 100%);
        padding: 30px 32px;
        border-radius: 20px;
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 8px 24px rgba(27, 94, 32, 0.28);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    .badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.18);
        backdrop-filter: blur(8px);
        padding: 4px 12px;
        border-radius: 30px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    .brand-title {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .brand-title span {
        color: #dcedc8;
        font-weight: 400;
        font-size: 1.5rem;
    }
    .greeting-line {
        margin: 10px 0 0 0;
        font-size: 1.15rem;
        font-weight: 600;
        opacity: 0.95;
    }
    .tagline-en {
        margin: 6px 0 0 0;
        font-size: 0.95rem;
        opacity: 0.9;
        line-height: 1.4;
    }
    .tagline-ta {
        margin-top: 10px;
        padding-top: 10px;
        border-top: 1px solid rgba(255, 255, 255, 0.2);
        font-size: 0.98rem;
        opacity: 0.92;
        font-weight: 500;
    }
    div.stButton > button {
        border-radius: 12px;
        border: 1px solid #a5d6a7;
        padding: 12px 10px;
        height: 100%;
    }
    div.stButton > button:hover {
        border-color: #2e7d32;
        color: #2e7d32;
        background: #e8f5e9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

AGENT_AVATAR = "🌾"
USER_AVATAR = "🧑‍🌾"
HISTORY_SIZE = 6


# ---------- Helpers ----------
def greeting():
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning", "காலை வணக்கம்", "🌅"
    if hour < 17:
        return "Good afternoon", "மதிய வணக்கம்", "☀️"
    return "Good evening", "மாலை வணக்கம்", "🌇"


def load_chats():
    try:
        with open(CHATS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_chats():
    try:
        with open(CHATS_FILE, "w", encoding="utf-8") as f:
            json.dump(st.session_state.chats, f, ensure_ascii=False)
    except Exception:
        pass


def current_messages():
    cid = st.session_state.current
    return st.session_state.chats[cid]["messages"] if cid in st.session_state.chats else []


def stream_from_api(question, history, sources_out):
    payload = {"question": question, "history": history}
    with requests.post(f"{API_URL}/chat/stream", json=payload, stream=True, timeout=90) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            ev = json.loads(line)
            if ev["type"] == "token":
                yield ev["data"]
            elif ev["type"] == "sources":
                sources_out.extend(ev["data"])
            elif ev["type"] == "error":
                raise RuntimeError(ev["data"])


# ---------- Session State ----------
if "chats" not in st.session_state:
    st.session_state.chats = load_chats()
if "current" not in st.session_state:
    st.session_state.current = None

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### 🌾 **TN-AGRI BOT**")
    st.caption("AI Scheme Advisor for Tamil Nadu Farmers")
    st.write("")

    if st.button("➕ New chat", use_container_width=True):
        st.session_state.current = None
        st.rerun()

    st.markdown("##### 🕘 Recent chats")
    chats = st.session_state.chats
    if not chats:
        st.caption("No chats yet. Ask your first question!")
    recent = sorted(chats.items(), key=lambda kv: kv[1]["created"], reverse=True)[:15]
    for cid, chat in recent:
        c1, c2 = st.columns([5, 1])
        is_current = cid == st.session_state.current
        label = ("👉 " if is_current else "💬 ") + chat["title"]
        if c1.button(label, key=f"open_{cid}", use_container_width=True):
            st.session_state.current = cid
            st.rerun()
        if c2.button("✕", key=f"del_{cid}"):
            del st.session_state.chats[cid]
            if is_current:
                st.session_state.current = None
            save_chats()
            st.rerun()

    st.divider()
    st.header("🌱 About")
    st.write("Answers questions about Tamil Nadu agriculture schemes using official scheme documents.")
    st.caption("Ask in English or தமிழ்")
    st.info(
        "Information comes from the scheme documents and may change. "
        "Please confirm with your local agriculture office before applying."
    )

# ---------- Hero Banner ----------
hello_en, hello_ta, icon = greeting()
st.markdown(
    f"""
    <div class="hero">
        <div class="badge">🌱 Tamil Nadu Agri AI Assistant</div>
        <div class="brand-title">🌾 TN-AGRI BOT <span>| உழவன் துணை</span></div>
        <div class="greeting-line">{icon} {hello_en}! / {hello_ta}!</div>
        <p class="tagline-en">Your personal digital assistant for Tamil Nadu Government agriculture schemes, subsidies, and farmer welfare benefits.</p>
        <div class="tagline-ta">விவசாய மானியங்கள், பயிர் காப்பீடு மற்றும் அரசு நலத்திட்டங்கள் குறித்த உடனடி வழிகாட்டி.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Show Current Chat ----------
msgs = current_messages()
for m in msgs:
    avatar = AGENT_AVATAR if m["role"] == "assistant" else USER_AVATAR
    with st.chat_message(m["role"], avatar=avatar):
        st.markdown(m["content"])
        if m.get("sources") and len(m["sources"]) > 0:
            st.caption("📚 Sources: " + ", ".join(m["sources"]))

# ---------- Suggestion Cards (Empty Chat Only) ----------
SUGGESTIONS = [
    ("💰 Subsidies", "What subsidies are available for small and marginal farmers?"),
    ("📄 Documents", "What documents are needed to apply for a scheme?"),
    ("✅ Eligibility", "Who is eligible for agriculture schemes?"),
    ("🌾 தமிழில் கேள்", "விவசாயிகளுக்கு என்னென்ன மானியங்கள் கிடைக்கும்?"),
]

if not msgs:
    st.markdown("##### 💡 Try asking")
    cols = st.columns(2)
    for i, (label, text) in enumerate(SUGGESTIONS):
        with cols[i % 2]:
            if st.button(f"{label}\n\n{text}", key=f"sug{i}", use_container_width=True):
                st.session_state.pending = text

# ---------- Handle User Question ----------
user_input = st.chat_input("Ask about a scheme...")
question = user_input or st.session_state.pop("pending", None)

if question:
    # First question of a new chat -> create the chat entry
    if st.session_state.current is None:
        cid = uuid.uuid4().hex[:8]
        clean_text = question.strip().split("\n")[-1]
        title = clean_text[:35] + ("…" if len(clean_text) > 35 else "")
        st.session_state.chats[cid] = {"title": title, "created": time.time(), "messages": []}
        st.session_state.current = cid

    msgs = current_messages()

    history = [{"role": m["role"], "content": m["content"]} for m in msgs[-HISTORY_SIZE:]]
    msgs.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(question)

    ok = False
    with st.chat_message("assistant", avatar=AGENT_AVATAR):
        sources = []
        try:
            gen = stream_from_api(question, history, sources)
            with st.spinner("🔍 Searching scheme documents..."):
                first = next(gen, "")
            reply = st.write_stream(itertools.chain([first], gen))
            msgs.append({"role": "assistant", "content": reply, "sources": sources})
            save_chats()
            ok = True
        except requests.exceptions.ConnectionError:
            st.error("Cannot reach the backend API. Start it with: uvicorn api:app --port 8000")
        except Exception as e:
            st.error(str(e) if isinstance(e, RuntimeError) else "Sorry, something went wrong. Please try again.")

    if ok:
        st.rerun()
    else:
        msgs.pop()
        if not msgs:
            st.session_state.chats.pop(st.session_state.current, None)
            st.session_state.current = None