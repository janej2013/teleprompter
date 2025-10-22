# Real-time Teleprompter Prototype

This repository contains a prototype real-time teleprompter that streams microphone audio to the OpenAI Realtime API, detects interview questions, and surfaces suggested answers on a teleprompter-style React frontend.

## Features

- **FastAPI backend** that captures audio frames, forwards them to the OpenAI Realtime API, and performs semantic search over prepared interview Q&A pairs using FAISS and `text-embedding-3-small` embeddings.
- **Question detection** using a simple heuristic to trigger answer lookup or GPT-5 fallback generation.
- **React + Vite frontend** that provides a teleprompter display with large typography, auto-scrolling containers, and a start/stop recording button.
- **WebSocket communication** between frontend and backend for streaming status updates, transcripts, and answers.

## Running locally

### Prerequisites

- Python 3.9+
- Node.js 18+
- An OpenAI API key with access to the Realtime and GPT-5 APIs.

### Backend setup

1. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Set environment variables (consider creating a `.env` file):

   ```bash
   export OPENAI_API_KEY="sk-..."
   export OPENAI_REALTIME_MODEL="gpt-4o-realtime-preview-2024-12-17"  # or another realtime-capable model
   ```

3. Start the FastAPI server:

   ```bash
   uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
   ```

### Frontend setup

1. Install Node dependencies:

   ```bash
   cd frontend
   npm install
   ```

2. Launch the development server:

   ```bash
   npm run dev
   ```

3. Open the provided URL (usually `http://localhost:5173`) in a Chromium-based browser. Grant microphone access when prompted.

### Using the prototype

- Click **Start Teleprompter** to begin streaming microphone audio to the backend. The backend forwards chunks to the OpenAI Realtime API for transcription.
- When the interviewer asks a question, the backend will surface matching answers from `backend/qa_data.json`. If no close match is found (similarity < 0.75), the backend requests a short GPT-5 answer.
- The suggested answer appears in the teleprompter section so you can read it aloud.

> **Note:** This is a prototype intended for demonstration purposes. Error handling and authentication have been kept intentionally simple.
