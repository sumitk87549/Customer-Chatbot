from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import google.generativeai as genai


"""setup API key"""

os.environ['GOOGLE_API_KEY'] = **********
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

print([m.name for m in genai.list_models()])
model = genai.GenerativeModel("gemini-flash-lite-latest")
chat = model.start_chat(history=[])

def get_gemini_response(question):
    response = chat.send_message(question, stream=True)
    return response

st.set_page_config(page_title="CHATBOT-DEMO")
st.header("GEMINI LLM APP")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
input = st.text_input("input: ", key="input")
submit = st.button("ASK THE QUESTION ")

if submit and input:
    response = get_gemini_response(input)
    st.session_state['chat_history'].append(("YOU", input))
    st.subheader("The response is ")
    for chunk in response:
        st.write(chunk.text)
        st.session_state['chat_history'].append(("BOT", chunk.text))
st.subheader("THE CHAT HISTORY IS ")
for role,text in st.session_state['chat_history']:
    st.write(f"{role}:{text}")

