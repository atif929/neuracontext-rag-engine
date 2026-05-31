import os
import shutil

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from generator import generate
from ingestion import chunk_text, ingest
from retriever import hybrid_retrieve, load_artifacts
from utils.parser import parse_pdf

app = FastAPI(title="AI Knowledge Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Vite default dev port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "/tmp/documents"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Startup: load index artifacts if they already exist on disk
# ---------------------------------------------------------------------------
_index = None
_bm25 = None
_chunks = None


def _try_load():
    global _index, _bm25, _chunks
    try:
        _index, _bm25, _chunks = load_artifacts()
    except FileNotFoundError:
        pass  # No documents uploaded yet — that's fine


_try_load()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class QueryResponse(BaseModel):
    response: str
    chunks_used: int


class UploadResponse(BaseModel):
    filename: str
    chunks_created: int
    message: str


# ---------------------------------------------------------------------------
# POST /upload
# ---------------------------------------------------------------------------
@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        text = parse_pdf(save_path)
    except ValueError as e:
        os.remove(save_path)
        raise HTTPException(status_code=422, detail=str(e))

    chunks = chunk_text(text, chunk_size=500, overlap=50)
    if not chunks:
        raise HTTPException(status_code=422, detail="Could not extract any text chunks from this PDF.")

    ingest(chunks)

    # Reload artifacts so queries immediately use the new document
    global _index, _bm25, _chunks
    _index, _bm25, _chunks = load_artifacts()

    return UploadResponse(
        filename=file.filename,
        chunks_created=len(chunks),
        message=f"Successfully ingested {len(chunks)} chunks from '{file.filename}'.",
    )


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------
@app.post("/query", response_model=QueryResponse)
async def query_documents(body: QueryRequest):
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    if _index is None:
        raise HTTPException(
            status_code=404,
            detail="No documents have been uploaded yet. Please POST a PDF to /upload first.",
        )

    try:
        retrieved_chunks = hybrid_retrieve(
            query=body.query,
            index=_index,
            bm25=_bm25,
            chunks=_chunks,
            top_k=body.top_k,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    response_text = generate(body.query, retrieved_chunks)

    return QueryResponse(
        response=response_text,
        chunks_used=len(retrieved_chunks),
    )


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "documents_loaded": _chunks is not None,
        "total_chunks": len(_chunks) if _chunks else 0,
    }
