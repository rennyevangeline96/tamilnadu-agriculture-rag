import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import rag

app = FastAPI(
    title="TN Agriculture Schemes RAG API",
    version="1.0",
    description=(
        "Backend for the Tamil Nadu Agriculture Schemes chatbot. "
        "Answers questions in English or Tamil using official scheme PDFs "
        "(FAISS retrieval + gpt-4o-mini + NeMo Guardrails), with PDF page sources. "
        "Requests are traced in LangSmith."
    ),
)


class Message(BaseModel):
    role: str = Field(description="Either 'user' or 'assistant'", examples=["user"])
    content: str = Field(description="The message text")


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        max_length=1500,
        description="The user's question, in English or Tamil",
        examples=["What subsidies are available for small farmers?"],
    )
    history: list[Message] = Field(
        default=[],
        description="Recent chat messages (memory). Leave empty for a first question.",
    )


class ChatResponse(BaseModel):
    answer: str = Field(description="The generated answer")
    sources: list[str] = Field(description="PDF names and pages the answer came from")


@app.get("/", tags=["General"], summary="API welcome message")
def root():
    """Returns a simple message confirming the API is running."""
    return {"message": "TN Agriculture Schemes RAG API is running. See /docs for the API docs."}


@app.get("/health", tags=["General"], summary="Health check")
def health():
    """Returns `{"status": "ok"}` if the server is alive. Use it to check the backend started."""
    return {"status": "ok"}


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Chat"],
    summary="Ask a question (full answer at once)",
)
def chat(req: ChatRequest):
    """
    Takes a question and optional chat history, searches the scheme documents,
    and returns the complete answer with sources in one JSON payload.
    Best for testing in Swagger or Postman.
    """
    history = [m.model_dump() for m in req.history]
    try:
        answer_parts = []
        sources = []
        for ev in rag.run_rag(req.question, history):
            if ev.get("type") == "token":
                answer_parts.append(ev.get("data", ""))
            elif ev.get("type") == "sources":
                sources = ev.get("data", [])
        return {"answer": "".join(answer_parts), "sources": sources}
    except Exception as e:
        return {"answer": rag.friendly_error(e), "sources": []}


@app.post(
    "/chat/stream",
    tags=["Chat"],
    summary="Ask a question (streamed answer, typing effect)",
)
def chat_stream(req: ChatRequest):
    """
    Streams the answer line-by-line as ndjson events:
    - `{"type": "token", "data": "..."}`
    - `{"type": "sources", "data": [...]}`
    - `{"type": "error", "data": "..."}`
    Used directly by the Streamlit frontend.
    """
    history = [m.model_dump() for m in req.history]

    def events():
        try:
            for ev in rag.run_rag(req.question, history):
                yield json.dumps(ev, ensure_ascii=False) + "\n"
        except Exception as e:
            error_msg = (
                rag.friendly_error(e)
                if hasattr(rag, "friendly_error")
                else "An unexpected error occurred."
            )
            yield json.dumps({"type": "error", "data": error_msg}, ensure_ascii=False) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")