import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

# Initialize model
model = init_chat_model("mistral-small-2603")

st.set_page_config(page_title="AI Mode Chat Bot", page_icon="🤖")
st.title("🤖 AI Chat Bot with Personality Modes")

# Mode selection UI
st.sidebar.title("Choose Your AI Mode")
choice = st.sidebar.radio(
    "Select Mode",
    ("Angry Mode", "Funny Mode", "Sad Mode")
)

# Define system prompt based on choice
if choice == "Angry Mode":
    mode = "You are an angry AI agent. You respond aggressively and impatiently."
elif choice == "Funny Mode":
    mode = "You are a funny AI agent. You respond with humor and jokes."
else:
    mode = "You are a sad AI agent. You respond in a sad tone."

# Initialize session memory
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content=mode)
    ]

# Reset memory if mode changes
if st.session_state.messages[0].content != mode:
    st.session_state.messages = [SystemMessage(content=mode)]

st.write("Type '0' to exit")

# Display chat history
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

# Chat input
user_input = st.chat_input("Type your message...")

if user_input:
    if user_input == "0":
        st.stop()

    # Store user message
    st.session_state.messages.append(HumanMessage(content=user_input))

    with st.chat_message("user"):
        st.write(user_input)

    # Get model response
    response = model.invoke(st.session_state.messages)

    # Store AI response
    st.session_state.messages.append(AIMessage(content=response.content))

    with st.chat_message("assistant"):
        st.write(response.content)