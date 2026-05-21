# ================================
# STREAMLIT UI - CITY INTELLIGENCE SYSTEM
# ================================

import streamlit as st
import os
import requests
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv()

# ================================
# PAGE CONFIG
# ================================

st.set_page_config(
    page_title="City Intelligence System",
    page_icon="🏙️",
    layout="wide"
)

# ================================
# CUSTOM CSS
# ================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

.stApp {
    background-color: #0a0a0f;
    color: #e8e8f0;
}

/* Header */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 3rem;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #c8f564 0%, #4de8c2 60%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0;
}

.hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #666688;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 2px;
}

/* Chat messages */
.msg-user {
    background: #1a1a2e;
    border-left: 3px solid #c8f564;
    padding: 14px 18px;
    border-radius: 0 10px 10px 0;
    margin: 10px 0;
    font-family: 'Syne', sans-serif;
    font-size: 0.95rem;
}

.msg-assistant {
    background: #0f1f2e;
    border-left: 3px solid #4de8c2;
    padding: 14px 18px;
    border-radius: 0 10px 10px 0;
    margin: 10px 0;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    line-height: 1.7;
    color: #b8d4e8;
}

.msg-tool {
    background: #1a0f2e;
    border-left: 3px solid #a78bfa;
    padding: 10px 16px;
    border-radius: 0 8px 8px 0;
    margin: 6px 0 6px 20px;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #a78bfa;
}

.label-user {
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #c8f564;
    font-family: 'Space Mono', monospace;
    margin-bottom: 4px;
}

.label-assistant {
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #4de8c2;
    font-family: 'Space Mono', monospace;
    margin-bottom: 4px;
}

.label-tool {
    font-size: 0.62rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #a78bfa;
    font-family: 'Space Mono', monospace;
    margin-bottom: 3px;
}

/* Tool approval box */
.approval-box {
    background: #1a1000;
    border: 1px solid #c8f564;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 12px 0;
}

.approval-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: #c8f564;
    letter-spacing: 1px;
    margin-bottom: 6px;
}

/* Divider */
hr {
    border-color: #1e1e3a;
}

/* Input override */
.stTextInput > div > div > input {
    background: #12121f !important;
    border: 1px solid #2a2a4a !important;
    border-radius: 8px !important;
    color: #e8e8f0 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 12px 16px !important;
}

.stTextInput > div > div > input:focus {
    border-color: #c8f564 !important;
    box-shadow: 0 0 0 2px rgba(200, 245, 100, 0.1) !important;
}

/* Buttons */
.stButton > button {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 1px !important;
    border-radius: 6px !important;
    padding: 8px 20px !important;
    transition: all 0.2s ease !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d0d1a;
    border-right: 1px solid #1e1e3a;
}

.sidebar-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 2px;
    color: #666688;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}

.dot-green { background: #c8f564; box-shadow: 0 0 6px #c8f564; }
.dot-red   { background: #ff4466; box-shadow: 0 0 6px #ff4466; }
</style>
""", unsafe_allow_html=True)


# ================================
# TOOLS (unchanged from original)
# ================================

@tool
def get_weather(city: str) -> str:
    """Get current weather of a city"""
    API_KEY = os.getenv("OPENWEATHER_API_KEY")
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={API_KEY}&units=metric"
    )
    response = requests.get(url)
    data = response.json()
    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Could not fetch weather')}"
    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]
    return f"Weather in {city}: {desc}, {temp}°C"


@tool
def get_news(city: str) -> str:
    """Get latest news about the city"""
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    query = f"Latest news about {city}"
    response = tavily_client.search(
        query=query,
        search_depth="basic",
        max_results=3
    )
    results = response.get("results", [])
    if not results:
        return f"No news found for {city}"
    news_list = []
    for item in results:
        title = item.get("title", "")
        content = item.get("content", "")
        url = item.get("url", "")
        news_list.append(
            f"Title: {title}\n"
            f"Summary: {content}\n"
            f"Source: {url}\n"
        )
    return "\n\n".join(news_list)


TOOLS_MAP = {
    "get_weather": get_weather,
    "get_news": get_news
}


# ================================
# SESSION STATE INIT
# ================================

if "messages" not in st.session_state:
    st.session_state.messages = []          # LangChain memory

if "chat_log" not in st.session_state:
    st.session_state.chat_log = []          # UI display log

if "pending_tool_calls" not in st.session_state:
    st.session_state.pending_tool_calls = []  # Tools awaiting approval

if "waiting_approval" not in st.session_state:
    st.session_state.waiting_approval = False

if "hitl_enabled" not in st.session_state:
    st.session_state.hitl_enabled = True


# ================================
# LLM INIT (cached)
# ================================

@st.cache_resource
def get_llm():
    llm = ChatMistralAI(model="mistral-tiny")
    return llm.bind_tools([get_weather, get_news])


llm_with_tool = get_llm()


# ================================
# HEADER
# ================================

col_title, col_clear = st.columns([5, 1])
with col_title:
    st.markdown('<div class="hero-title">🏙️ City Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Powered by Mistral · Tavily · OpenWeather</div>', unsafe_allow_html=True)

with col_clear:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑 Clear", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_log = []
        st.session_state.pending_tool_calls = []
        st.session_state.waiting_approval = False
        st.rerun()

st.markdown("<hr>", unsafe_allow_html=True)


# ================================
# SIDEBAR
# ================================

with st.sidebar:
    st.markdown('<div class="sidebar-label">Settings</div>', unsafe_allow_html=True)

    st.session_state.hitl_enabled = st.toggle(
        "Human-in-the-Loop",
        value=st.session_state.hitl_enabled,
        help="Require your approval before each tool call"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">API Status</div>', unsafe_allow_html=True)

    openweather_ok = bool(os.getenv("OPENWEATHER_API_KEY"))
    tavily_ok      = bool(os.getenv("TAVILY_API_KEY"))
    mistral_ok     = bool(os.getenv("MISTRAL_API_KEY"))

    def status_row(label, ok):
        dot = "dot-green" if ok else "dot-red"
        status = "Connected" if ok else "Missing"
        st.markdown(
            f'<span class="status-dot {dot}"></span>'
            f'<span style="font-family:Space Mono,monospace;font-size:0.78rem;color:#aaa">'
            f'{label}: <b style="color:{("#c8f564" if ok else "#ff4466")}">{status}</b></span>',
            unsafe_allow_html=True
        )

    status_row("OpenWeather", openweather_ok)
    status_row("Tavily", tavily_ok)
    status_row("Mistral", mistral_ok)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">Available Tools</div>', unsafe_allow_html=True)
    st.markdown(
        '<span style="font-family:Space Mono,monospace;font-size:0.78rem;color:#a78bfa">⚡ get_weather</span><br>'
        '<span style="font-family:Space Mono,monospace;font-size:0.78rem;color:#a78bfa">📰 get_news</span>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">Try asking</div>', unsafe_allow_html=True)
    example_prompts = [
        "What's the weather in Mumbai?",
        "Latest news about Delhi",
        "Weather and news for Ranchi",
    ]
    for ep in example_prompts:
        if st.button(ep, use_container_width=True, key=f"ex_{ep}"):
            st.session_state["prefill_input"] = ep
            st.rerun()


# ================================
# RENDER CHAT LOG
# ================================

chat_container = st.container()

with chat_container:
    for entry in st.session_state.chat_log:
        if entry["role"] == "user":
            st.markdown(f'<div class="label-user">You</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="msg-user">{entry["content"]}</div>', unsafe_allow_html=True)

        elif entry["role"] == "assistant":
            st.markdown(f'<div class="label-assistant">Agent</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="msg-assistant">{entry["content"]}</div>', unsafe_allow_html=True)

        elif entry["role"] == "tool":
            st.markdown(f'<div class="label-tool">⚡ Tool · {entry["tool_name"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="msg-tool">{entry["content"]}</div>', unsafe_allow_html=True)


# ================================
# TOOL APPROVAL UI (Human-in-the-Loop)
# ================================

if st.session_state.waiting_approval and st.session_state.pending_tool_calls:
    tool_call = st.session_state.pending_tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call.get("args", {})

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        f'<div class="approval-box">'
        f'<div class="approval-title">🔐 Tool Approval Required</div>'
        f'<span style="font-family:Space Mono,monospace;font-size:0.82rem;color:#e8e8f0">'
        f'Agent wants to call <b style="color:#c8f564">{tool_name}</b> '
        f'with args: <b style="color:#4de8c2">{tool_args}</b>'
        f'</span></div>',
        unsafe_allow_html=True
    )

    col_yes, col_no, _ = st.columns([1, 1, 4])
    with col_yes:
        if st.button("✅ Approve", use_container_width=True, key="approve_btn"):
            # Execute approved tool
            tool_result = TOOLS_MAP[tool_name].invoke(tool_call)

            st.session_state.chat_log.append({
                "role": "tool",
                "tool_name": tool_name,
                "content": tool_result
            })

            st.session_state.messages.append(
                ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call["id"]
                )
            )

            st.session_state.pending_tool_calls.pop(0)

            # If more pending tools exist, stay in approval mode
            if st.session_state.pending_tool_calls:
                st.rerun()
            else:
                st.session_state.waiting_approval = False
                # Continue agent loop
                st.session_state["continue_agent"] = True
                st.rerun()

    with col_no:
        if st.button("❌ Deny", use_container_width=True, key="deny_btn"):
            st.session_state.chat_log.append({
                "role": "assistant",
                "content": "Tool call denied. Cannot fetch latest information."
            })
            st.session_state.pending_tool_calls = []
            st.session_state.waiting_approval = False
            st.rerun()


# ================================
# CONTINUE AGENT AFTER TOOL RESULT
# ================================

if st.session_state.get("continue_agent") and not st.session_state.waiting_approval:
    st.session_state["continue_agent"] = False

    with st.spinner("Agent is thinking..."):
        while True:
            result = llm_with_tool.invoke(st.session_state.messages)
            st.session_state.messages.append(result)

            if result.tool_calls:
                if st.session_state.hitl_enabled:
                    st.session_state.pending_tool_calls = result.tool_calls
                    st.session_state.waiting_approval = True
                    st.rerun()
                    break
                else:
                    # Auto-execute without approval
                    for tc in result.tool_calls:
                        tool_result = TOOLS_MAP[tc["name"]].invoke(tc)
                        st.session_state.chat_log.append({
                            "role": "tool",
                            "tool_name": tc["name"],
                            "content": tool_result
                        })
                        st.session_state.messages.append(
                            ToolMessage(content=tool_result, tool_call_id=tc["id"])
                        )
                    continue
            else:
                st.session_state.chat_log.append({
                    "role": "assistant",
                    "content": result.content
                })
                break

    st.rerun()


# ================================
# INPUT BAR
# ================================

st.markdown("<br>", unsafe_allow_html=True)

prefill = st.session_state.pop("prefill_input", "")

with st.form(key="chat_form", clear_on_submit=True):
    col_input, col_send = st.columns([5, 1])
    with col_input:
        user_input = st.text_input(
            label="",
            placeholder="Ask about weather or news for any city...",
            value=prefill,
            label_visibility="collapsed"
        )
    with col_send:
        submitted = st.form_submit_button("Send →", use_container_width=True)


# ================================
# HANDLE SUBMISSION
# ================================

if submitted and user_input.strip():
    user_text = user_input.strip()

    st.session_state.chat_log.append({"role": "user", "content": user_text})
    st.session_state.messages.append(HumanMessage(content=user_text))

    with st.spinner("Agent is thinking..."):
        while True:
            result = llm_with_tool.invoke(st.session_state.messages)
            st.session_state.messages.append(result)

            if result.tool_calls:
                if st.session_state.hitl_enabled:
                    st.session_state.pending_tool_calls = result.tool_calls
                    st.session_state.waiting_approval = True
                    break
                else:
                    # Auto-execute all tools without approval
                    for tc in result.tool_calls:
                        tool_result = TOOLS_MAP[tc["name"]].invoke(tc)
                        st.session_state.chat_log.append({
                            "role": "tool",
                            "tool_name": tc["name"],
                            "content": tool_result
                        })
                        st.session_state.messages.append(
                            ToolMessage(content=tool_result, tool_call_id=tc["id"])
                        )
                    continue
            else:
                st.session_state.chat_log.append({
                    "role": "assistant",
                    "content": result.content
                })
                break

    st.rerun()