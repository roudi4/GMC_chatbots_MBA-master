import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_classic.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from PyPDF2 import PdfReader


BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "src" / "template"
DATA_DIR = BASE_DIR / "src" / "data"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_MODEL = "gemini-2.5-flash"


load_dotenv()

app = FastAPI(title="ISG RAG Chatbot API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def read_pdf_text(pdf_paths: list[Path]) -> str:
    parts = []
    for pdf_path in pdf_paths:
        pdf_reader = PdfReader(str(pdf_path))
        for page in pdf_reader.pages:
            parts.append(page.extract_text() or "")
    return "".join(parts)


def split_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
    return splitter.split_text(text)


def build_vector_store(pdf_paths: list[Path]) -> None:
    text = read_pdf_text(pdf_paths)
    chunks = split_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No readable text was found in the uploaded PDFs.")

    vector_store = FAISS.from_texts(chunks, embedding=get_embeddings())
    vector_store.save_local(str(VECTOR_STORE_DIR))


def get_conversational_chain():
    prompt_template = """answer the question as detailed as possible from the provided context,
make sure to provide all the details, if the answer is not in the provided context,
just say "I am sorry, answer is not available in this context" and do not provide a wrong answer.

Context: {context}

Question: {question}
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    model = ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=os.getenv("GOOGLE_API_KEY"))
    return load_qa_chain(llm=model, prompt=prompt)


def ensure_vector_store_exists() -> None:
    if not (VECTOR_STORE_DIR / "index.faiss").exists():
        raise HTTPException(
            status_code=400,
            detail="No documents have been processed yet. Please upload PDFs first.",
        )


def process_user_question(question: str) -> str:
    ensure_vector_store_exists()
    vector_store = FAISS.load_local(
        str(VECTOR_STORE_DIR),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )
    docs = vector_store.similarity_search(question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": question},
        return_only_outputs=True,
    )
    return response["output_text"]


def load_existing_documents() -> None:
    if (VECTOR_STORE_DIR / "index.faiss").exists() or not DATA_DIR.exists():
        return

    pdf_files = list(DATA_DIR.glob("*.pdf"))
    if pdf_files:
        build_vector_store(pdf_files)


@app.post("/api/chat")
async def chat(request: ChatRequest):
    return {"answer": process_user_question(request.question)}


@app.post("/api/upload")
async def upload_pdfs(files: list[UploadFile] = File(...)):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pdf_paths: list[Path] = []

    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            continue

        file_path = DATA_DIR / file.filename
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        pdf_paths.append(file_path)

    if not pdf_paths:
        raise HTTPException(status_code=400, detail="No valid PDF files uploaded.")

    build_vector_store(pdf_paths)
    return {
        "status": "success",
        "message": "PDFs processed successfully.",
        "files_uploaded": len(pdf_paths),
    }


@app.on_event("startup") # Load existing vector store or process PDFs in data directory on startup
# This ensures that the chatbot is ready to answer questions immediately after the server starts, without requiring a new upload each time.
async def startup_event(): 
    load_existing_documents()# Load existing vector store or process PDFs in data directory on startup


app.mount("/", StaticFiles(directory=str(TEMPLATE_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
