# ISG AI Chatbot Project

This repository contains 3 different AI applications built for ISG project:

---

## Applications

### 1. `app1.py` - Simple Gemini Chat Interface
**Basic Chatbot Application**
- Simple Streamlit web interface
- Direct conversation with Google Gemini AI
- No context, no documents required
- Single question/answer interface

**Run command:**
```bash
streamlit run app1.py
```

---

### 2. `rag_app2.py` - RAG Document Chatbot
**Retrieval Augmented Generation Application**
- Upload your own PDF documents
- Extract text and create vector embeddings
- Ask questions about content inside your documents
- Answers only from provided document context
- Local FAISS vector store

**Run command:**
```bash
streamlit run rag_app2.py
```

**Features:**
- Multiple PDF upload support
- Automatic text chunking
- Vector database storage
- Context-aware answers

---

### 3. `appApi3.py` - Full Web API + Frontend
**Complete Production Grade Application**
- FastAPI backend server
- REST API endpoints
- Built-in CORS support
- Web frontend UI (HTML/CSS/JS)
- Automatic PDF processing on startup
- Persistent document storage

**Run command:**
```bash
python appApi3.py
```

**Access:**
- Web Interface: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`

**API Endpoints:**
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/chat` | Send question and get answer |
| POST | `/api/upload` | Upload PDF documents |
| GET | `/` | Web frontend interface |

---

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment Setup
1. Copy `.env.example` to `.env`
2. Add your Google API Key:
```env
GOOGLE_API_KEY=your_api_key_here
```

### 3. Get Google API Key
1. Go to https://makersuite.google.com/app/apikey
2. Create new API key
3. Paste in `.env` file

---

## Project Structure
```
ISG_bot/
├── app1.py              # Simple Gemini Chat
├── rag_app2.py          # RAG Streamlit App
├── appApi3.py           # FastAPI Full Application
├── requirements.txt     # Dependencies
├── .env.example         # Environment template
├── .env                 # Your actual environment variables
└── src/
    ├── data/            # Stored PDF documents
    └── template/        # Web frontend files
        ├── index.html
        ├── style.css
        └── app.js
```

---

## Usage Guide

### First Time Run:
1. Complete setup steps above
2. Choose which application you want to run
3. Run the corresponding command
4. Open browser at shown address

### Order of Complexity:
1. **app1.py** Simplest - for testing basic AI connection
2. **rag_app2.py** Intermediate - for testing RAG functionality
3. **appApi3.py** Advanced - full application with API and web UI

---

## Notes
- All applications require active internet connection
- Google Gemini API has rate limits
- Vector store is stored locally on your machine
- Uploaded PDFs are saved in `src/data/` directory
- Uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings
- Uses Gemini 2.5 Flash model for generation

---

## Troubleshooting

- **Missing API Key**: Make sure `.env` file exists with correct GOOGLE_API_KEY
- **Dependencies Error**: Run `pip install --upgrade -r requirements.txt`
- **Port 8000 busy**: Change port in `appApi3.py` last line
- **PDF Loading issues**: Ensure PDF files are not encrypted or corrupted

---
###### Developed by Montasar bellah Abdallah