# AI Knowledge Assistant — RAG Chatbot
# NeuraContext RAG Engine

RAG chatbot with hybrid BM25 + FAISS search.
A hybrid Retrieval-Augmented Generation (RAG) system for intelligent document understanding and question answering over PDF files. The system combines semantic vector search and lexical retrieval to deliver accurate, context-grounded responses using large language models.

## Quick start
---

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
## Overview

NeuraContext RAG Engine is designed to extract knowledge from uploaded documents and enable natural language querying over them. It implements a full RAG pipeline consisting of document ingestion, chunking, embedding generation, hybrid retrieval (dense + sparse), rank fusion, and LLM-based response generation.

The system is optimized for accuracy, relevance, and scalability, using a combination of FAISS vector search and BM25 keyword matching, fused via Reciprocal Rank Fusion (RRF).

---

## Features

- PDF document upload and parsing
- Overlapping chunk-based text segmentation
- Semantic search using Sentence Transformers
- Lexical search using BM25 ranking
- Hybrid retrieval using FAISS + BM25 fusion (RRF)
- Context-aware response generation using LLM (Groq API)
- Persistent vector storage with FAISS indexing
- Interactive chat-based UI for querying documents
- Clean API layer built with FastAPI

---

## System Architecture

Frontend (React + Tailwind)
        ↓
FastAPI Backend
        ↓
Document Ingestion Pipeline
(PDF → Text → Chunking)
        ↓
Embedding Layer
(Sentence Transformers)
        ↓
Vector Store + Indexing
(FAISS + Stored Chunks)
        ↓
Hybrid Retrieval Engine
FAISS (Dense Search)
BM25 (Sparse Search)
        ↓
Reciprocal Rank Fusion (RRF)
        ↓
Context Builder
        ↓
LLM (Groq / LLaMA 3)
        ↓
Final Response

---

## Core Components

### 1. Document Ingestion
Uploaded PDFs are parsed into raw text and split into overlapping chunks to preserve contextual continuity.

### 2. Embedding & Vector Store
Chunks are converted into embeddings using `all-MiniLM-L6-v2` and stored in FAISS for fast similarity search.

### 3. Sparse Retrieval (BM25)
BM25 is used for keyword-based retrieval to improve recall on exact term queries.

### 4. Hybrid Retrieval (RRF)
FAISS and BM25 rankings are fused using Reciprocal Rank Fusion to produce a unified relevance ranking.

### 5. LLM Generation
Retrieved context is passed to a Groq-hosted LLaMA 3 model. The model generates responses strictly grounded in retrieved context.

---

## Tech Stack

Backend:
- FastAPI
- Python
- FAISS
- SentenceTransformers
- rank-bm25

Frontend:
- React
- Tailwind CSS

LLM:
- Groq API (LLaMA 3)

---

## API Endpoints

### POST /upload
Uploads and processes a PDF file.

Response:
{
  "message": "File processed successfully",
  "chunks_created": 42
}

---

### POST /query
Sends a query and returns a grounded response.

Request:
{
  "query": "What is this document about?",
  "top_k": 5
}

Response:
{
  "response": "..."
}

---

## Installation

Backend:
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
Create .env:
GROQ_API_KEY=your_api_key_here

Run backend:
uvicorn main:app --reload

---

Frontend:
npm install
npm run dev
```

Open http://localhost:5173 — upload a PDF, then start asking questions.
---

## Key Design Decisions

- Hybrid retrieval improves accuracy over pure vector search
- RRF reduces ranking bias between BM25 and FAISS
- Chunk overlap preserves context continuity
- Strict prompting reduces hallucinations
- Modular pipeline enables scalability

---

## Future Improvements

- Streaming responses
- Multi-document support
- User authentication
- Cloud deployment
- Analytics dashboard

---

## License

For educational and portfolio use only.

---

## API
| Method | Path | Description |
|--------|------|-------------|
| POST | /upload | Upload a PDF; returns chunk count |
| POST | /query | Send a question; returns grounded answer |
| GET  | /health | Check index status |
## Author

## Adding an LLM
Open `backend/generator.py` and uncomment the Anthropic block (or adapt for OpenAI).
Set `ANTHROPIC_API_KEY` in your environment before starting the server.
Built as a full-stack AI system demonstrating modern RAG architecture, vector databases, and LLM integration.