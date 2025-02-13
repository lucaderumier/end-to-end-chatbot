import streamlit as st
import requests

# FastAPI server URL
API_URL = "http://127.0.0.1:8000"  # Change this if deployed

st.title("Chatbot 🤖")

# Session state for chat history
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    role = "You" if message["role"] == "user" else "Bot"
    st.chat_message(role).write(message["content"])

# User input field
user_input = st.chat_input("Ask me anything...")

if user_input:
    # Add user message to chat history and display it directly
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("You").write(user_input)

    # Detect if the session is new or not
    if st.session_state.session_id is None:
        # Start a new chat session
        response = requests.post(f"{API_URL}/chat/start", json={"user_input": user_input})
        if response.status_code == 200:
            data = response.json()
            st.session_state.session_id = data.get("session_id")
            bot_response = data.get("response")
            st.session_state.messages.append({"role": "bot", "content": bot_response})
            st.chat_message("Bot").write(bot_response)
        else:
            st.error("Failed to start chat session.")
    else:
        # Continue an existing chat
        response = requests.post(
            f"{API_URL}/chat/{st.session_state.session_id}/continue",
            json={"user_input": user_input}
        )
        if response.status_code == 200:
            data = response.json()
            bot_response = data.get("response")
            st.session_state.messages.append({"role": "bot", "content": bot_response})
            st.chat_message("Bot").write(bot_response)
        else:
            st.error("Failed to continue chat.")