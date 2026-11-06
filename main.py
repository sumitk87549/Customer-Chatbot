from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
import streamlit as st
import os

""" Prompt Template """

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "you are a jolly software engineer(particularly Full-stack developer and AI Engineer) and a helpful human. provide clear response to any relevant query"
        ),
        (
            "user",
            "Question: {question}"
        ),
    ]
)

""" Streamlit Frontend """

st.title("Langchain demo with Ollama- Llama")

input_text = st.text_input("Enter a topic... ")

""" Ollama AI """
llm = OllamaLLM(model = "llama3.2:3b")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser
if input_text:
    st.write(chain.invoke({"question": input_text}))