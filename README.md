# AI Knowledge Assistant — RAG Chatbot

RAG chatbot with hybrid BM25 + FAISS search.

## Quick start

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 — upload a PDF, then start asking questions.

## API
| Method | Path | Description |
|--------|------|-------------|
| POST | /upload | Upload a PDF; returns chunk count |
| POST | /query | Send a question; returns grounded answer |
| GET  | /health | Check index status |

## Adding an LLM
Open `backend/generator.py` and uncomment the Anthropic block (or adapt for OpenAI).
Set `ANTHROPIC_API_KEY` in your environment before starting the server.