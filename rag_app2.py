import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.genai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_classic.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file!")
client = genai.Client(api_key=GOOGLE_API_KEY)

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store=FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local("vector_store")

def get_conversational_chain():
    prompt_template="""answer the question as detailed as possible from the provided context,
    make sure to provide all the details,if the answer is not in the provided context,
    just say "I am sorry, answer is not avaible in this context" don't provide a wrong answer\n\n
    Context: {context} \n\n
    Quetion: {question} \n\n"""

    model=ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
    prompt=PromptTemplate( template=prompt_template,input_variables=["context","question"])
    chain=load_qa_chain(llm=model, prompt=prompt)
    return chain


def user_input(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    new_db=FAISS.load_local("vector_store",embeddings,allow_dangerous_deserialization=True)# allow_dangerous_deserialization=True is used to avoid errors during loading
    docs = new_db.similarity_search(user_question)
    chain=get_conversational_chain()
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("", response["output_text"])

def main():
    st.set_page_config(page_title="Question Answering")
    st.header("ask any information from your docs")
    user_question= st.text_input("enter your question")
    if user_question:
        user_input(user_question)
    with st.sidebar:
        st.title("menu:") 
        pdf_docs=st.file_uploader("upload your pdf files",accept_multiple_files=True) 
        if st.button("submit"):
            with st.spinner("Processing PDFs and creating vector store..."):
                text=get_pdf_text(pdf_docs)
                chunks=get_text_chunks(text)
                get_vector_store(chunks)
            st.success("Done! PDFs processed successfully.")

if __name__ == "__main__":
    main()      
    
#python -m streamlit run rag_app.py
