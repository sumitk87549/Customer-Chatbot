import streamlit as st
from langchain_ollama import ChatOllama

"Initialize AI"

ollama = ChatOllama(model="llama3.2:3b")

st.title( "🤖Llama3.2 🍠tiny-brain🧠" )

if "messages" not in st.session_state:
    st.session_state['messages'] = [
        {
            "role":"assistant", "content":"how can I help you?"
        }
    ]

"""Message History"""
for msg in st.session_state.messages:
    st.chat_message(
        msg['role'], avatar='🧑‍💻' if msg['role'] =='user' else '🐭'
    ).write(msg['content'])

"""generate response"""
def generate_response():
    messages = [
        {'role':msg['role'], 'content': msg['content']}
        for msg in st.session_state.messages
    ]
    prompt = messages[-1]['content']
    response = ollama.stream(messages)
    full_response = ""
    for token in response:
        full_response += token.content
        yield token.content
    st.session_state['full_messages'] = full_response 

if prompt := st.chat_input():
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user",avatar='🧑‍💻').write(prompt)

    with st.chat_message("assistant", avatar='🐭'):
        response = st.write_stream(generate_response())

    st.session_state.messages.append(
        {"role":"assistant", "content": st.session_state['full_messages']}
    )