from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import pathlib
import textwrap
from PIL import Image

import google.generativeai as genai

from IPython.display import display, Markdown

os.environ['GOOGLE_API_KEY'] = '********'
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

def get_gemini_response(input, image):
    model = genai.GenerativeModel("gemini-3.8-flash")
    if input != "":
        response = model.generate_content([input, image])
    else:
        response = model.generate_content(image)
    return response.text

st.set_page_config(page_title="GEMINI IMAGE")
st.header("GEMINI APP")
input = st.text_input("Input Prompt: ", key="input")
uploaded_file = st.file_uploader("choose an image...", type=['jpg','png','jpeg'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image Preview...", use_column_width=True)

submit = st.button("Tell me about the image")

if submit:
    response = get_gemini_response(input, image)
    st.subheader("Response is:")
    st.write(response)