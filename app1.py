from dotenv import load_dotenv
import os
import google.genai as genai
import streamlit as st

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(page_title="Gemini Chat")
st.header("Ask Gemini")

input = st.text_input("Enter your prompt:")
if st.button("Send") and input:
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=input
    )
    st.write(response.text)