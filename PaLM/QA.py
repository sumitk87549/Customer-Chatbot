from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import google.generativeai as genai

"API Key"
os.environ["GOOGLE_API_KEY"] = "************"

genai.configure(api_key = os.environ["GOOGLE_API_KEY"])

model = genai.GenerativeModel("gemini-3.8-flash")

chat = model.start_chat(history=[])

def get_gemini_response(question):
    respose = chat.send_message(question, stream=True)
    return respose

st.set_page_config(page_title="ChatBOT")
st.header("Gemini LLM APP (PaLM)")

if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = []

input = st.text_input("input: ", key="input")
submit = st.button("ASK IT...")

if submit and input:
    response = get_gemini_response(input)
    st.session_state['chat_history'].append(("YOU",input))
    st.subheader("Response:")
    for chunk in response:
        st.write(chunk.text)
        st.session_state['chat_history'].append(("BOT",chunk.text))

st.subheader("Chat History :-")
for role, text in st.session_state['chat_history']:
    st.write(f"{role}:{text}")