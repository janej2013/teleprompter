"""FastAPI backend for real-time voice-to-text teleprompter."""
import asyncio
import base64
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss                   # type: ignore
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI

# Configure logging for easier debugging and tracing of realtime events
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the FastAPI app and configure CORS so that the React frontend can connect.
app = FastAPI(title="Realtime Teleprompter API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazily instantiated OpenAI async client. Requires OPENAI_API_KEY to be set.
openai_client = AsyncOpenAI()

# Path to the prepared Q&A file. For a production application this could be replaced
# with a database lookup. For this prototype we load it once at startup.
DATA_PATH = Path(__file__).parent / "qa_data.json"


class AnswerIndex:
    """Helper that stores Q&A embeddings inside a FAISS index."""

    def __init__(self, qa_pairs: List[Dict[str, str]], embedding_model: str = "text-embedding-3-small"):
        self.qa_pairs = qa_pairs
        self.embedding_model = embedding_model
        self.index: Optional[faiss.IndexFlatIP] = None
        self.vectors: Optional[np.ndarray] = None
        self._build_index()

    async def _embed(self, texts: List[str]) -> np.ndarray:
        """Compute embeddings for the provided texts using OpenAI embeddings."""
        response = await openai_client.embeddings.create(model=self.embedding_model, input=texts)
        embeddings = np.array([item.embedding for item in response.data], dtype="float32")
        # Normalize the vectors so that cosine similarity is equivalent to inner product.
        faiss.normalize_L2(embeddings)
        return embeddings

    def _build_index_sync(self) -> None:
        """Blocking helper used during app startup."""
        loop = asyncio.get_event_loop()
        embeddings = loop.run_until_complete(self._embed([item["question"] for item in self.qa_pairs]))
        self.vectors = embeddings
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        logger.info("Loaded %d prepared answers into FAISS index", len(self.qa_pairs))

    def _build_index(self) -> None:
        # When running inside uvicorn, the default loop policy may be set after import, so
        # we defensively build the index lazily in the startup event.
        try:
            self._build_index_sync()
        except RuntimeError:
            # If no event loop exists yet, defer building until startup.
            pass

    async def ensure_index(self) -> None:
        """Called during application startup to guarantee the FAISS index exists."""
        if self.index is None:
            embeddings = await self._embed([item["question"] for item in self.qa_pairs])
            self.vectors = embeddings
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(embeddings)
            logger.info("Loaded %d prepared answers into FAISS index", len(self.qa_pairs))

    async def search(self, question: str) -> Tuple[Optional[Dict[str, str]], float]:
        """Return the best matching Q&A pair and its similarity score."""
        if self.index is None:
            await self.ensure_index()
        assert self.index is not None
        embedding = await self._embed([question])
        distances, indices = self.index.search(embedding, k=1)
        top_score = float(distances[0][0])
        top_index = int(indices[0][0])
        if top_index == -1:
            return None, top_score
        return self.qa_pairs[top_index], top_score


async def load_qa_pairs() -> List[Dict[str, str]]:
    """Load the prepared Q&A pairs from disk."""
    with DATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


@app.on_event("startup")
async def startup_event() -> None:
    """Load data and prepare FAISS index during application startup."""
    global answer_index
    qa_pairs = await load_qa_pairs()
    answer_index = AnswerIndex(qa_pairs)
    await answer_index.ensure_index()


async def generate_answer(prompt: str) -> str:
    """Fallback for when the semantic search does not find a strong enough match."""
    logger.info("Falling back to GPT-5 response for prompt: %s", prompt)
    response = await openai_client.responses.create(
        model="gpt-5.0-mini",  # Placeholder for the GPT-5 model name.
        input=(
            "You are a concise assistant helping someone answer interview questions. "
            "Respond in 2 to 3 sentences.\n\nQuestion: " + prompt
        ),
        max_output_tokens=200,
    )
    # The responses API returns a structured payload; concatenate the text segments.
    text_chunks = []
    for item in response.output:
        if item.type == "output_text":
            text_chunks.append(item.text)
    return "".join(text_chunks).strip()


async def choose_answer(question: str) -> str:
    """Select the best answer using semantic search with a GPT fallback."""
    best_match, score = await answer_index.search(question)
    logger.info("Top semantic similarity score: %.3f", score)
    if best_match and score >= 0.75:
        return best_match["answer"]
    return await generate_answer(question)


async def connect_realtime_session() -> "WebSocketClientProtocol":
    """Create a websocket session with the OpenAI Realtime API."""
    import websockets

    api_key = os.environ["OPENAI_API_KEY"]
    realtime_model = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview-2024-12-17")
    url = f"wss://api.openai.com/v1/realtime?model={realtime_model}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Beta": "realtime=v1",
    }

    return await websockets.connect(url, extra_headers=headers, ping_interval=10, ping_timeout=10)


async def forward_audio_to_realtime(realtime_ws, audio_base64: str) -> None:
    """Forward microphone audio frames from the frontend to the Realtime API."""
    audio_bytes = base64.b64decode(audio_base64)
    message = {
        "type": "input_audio_buffer.append",
        "audio": base64.b64encode(audio_bytes).decode("utf-8"),
    }
    await realtime_ws.send(json.dumps(message))


async def trigger_transcription(realtime_ws) -> None:
    """Ask the Realtime API to create a transcription response from buffered audio."""
    message = {
        "type": "response.create",
        "response": {
            "modalities": ["text"],
            "instructions": "Transcribe the speaker's latest utterance.",
        },
    }
    await realtime_ws.send(json.dumps(message))


@app.websocket("/ws")
async def websocket_endpoint(socket: WebSocket) -> None:
    """Main realtime endpoint used by the React client."""
    await socket.accept()
    realtime_ws = None
    transcript_buffer = ""
    try:
        while True:
            message = await socket.receive_json()
            msg_type = message.get("type")

            if msg_type == "control":
                action = message.get("action")
                if action == "start":
                    # Lazily connect to the OpenAI Realtime API when recording starts.
                    if realtime_ws is None:
                        realtime_ws = await connect_realtime_session()
                        logger.info("Connected to OpenAI Realtime session")
                    await socket.send_json({"type": "status", "message": "Recording started"})
                elif action == "stop":
                    if realtime_ws is not None:
                        await trigger_transcription(realtime_ws)
                    await socket.send_json({"type": "status", "message": "Recording stopped"})
                continue

            if msg_type == "audio_chunk" and realtime_ws is not None:
                await forward_audio_to_realtime(realtime_ws, message["data"])
                continue

            if msg_type == "transcription_request" and realtime_ws is not None:
                await trigger_transcription(realtime_ws)
                continue

            if msg_type == "poll_transcript" and realtime_ws is not None:
                # Retrieve all pending messages from the realtime websocket.
                try:
                    while True:
                        payload = await asyncio.wait_for(realtime_ws.recv(), timeout=0.1)
                        data = json.loads(payload)
                        event_type = data.get("type")

                        if event_type == "response.output_text.delta":
                            transcript_buffer += data.get("delta", "")
                        elif event_type == "response.completed":
                            logger.info("Transcript completed: %s", transcript_buffer)
                            if transcript_buffer.strip():
                                await handle_transcript(socket, transcript_buffer.strip())
                            transcript_buffer = ""
                        elif event_type == "error":
                            logger.error("Realtime error: %s", data)
                except asyncio.TimeoutError:
                    pass
    except WebSocketDisconnect:
        logger.info("Frontend disconnected")
    finally:
        if realtime_ws is not None:
            await realtime_ws.close()


async def handle_transcript(socket: WebSocket, transcript: str) -> None:
    """Detect whether the transcript is a question and send an answer."""
    await socket.send_json({"type": "transcript", "text": transcript})
    if is_question(transcript):
        answer = await choose_answer(transcript)
        await socket.send_json({"type": "answer", "text": answer})


def is_question(text: str) -> bool:
    """Simple heuristic for determining whether the interviewer asked a question."""
    lowered = text.strip().lower()
    return lowered.endswith("?") or lowered.split(" ")[0] in {"what", "why", "how", "when", "where", "who", "is", "are", "do", "does", "can"}


@app.get("/")
async def root() -> Dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
