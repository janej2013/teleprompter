import { useEffect, useRef, useState } from 'react';
import './styles.css';

// WebSocket endpoint for the FastAPI backend.
const WS_URL = (import.meta.env.VITE_WS_URL) || `ws://${window.location.hostname}:8000/ws`;

export default function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [status, setStatus] = useState('Idle');
  const [transcript, setTranscript] = useState('');
  const [answer, setAnswer] = useState('');
  const wsRef = useRef(null);
  const pollTimer = useRef(null);

  useEffect(() => {
    return () => {
      stopRecording();
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const ensureSocket = () => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      const socket = new WebSocket(WS_URL);
      socket.onopen = () => {
        setStatus('Connected');
      };
      socket.onmessage = (event) => {
        const payload = JSON.parse(event.data);
        switch (payload.type) {
          case 'status':
            setStatus(payload.message);
            break;
          case 'transcript':
            setTranscript(payload.text);
            break;
          case 'answer':
            setAnswer(payload.text);
            break;
          default:
            break;
        }
      };
      socket.onclose = () => {
        setStatus('Disconnected');
        setIsRecording(false);
        if (pollTimer.current) {
          clearInterval(pollTimer.current);
          pollTimer.current = null;
        }
      };
      wsRef.current = socket;
    }
    return wsRef.current;
  };

  const startRecording = async () => {
    const socket = ensureSocket();
    if (!socket) return;
    if (socket.readyState !== WebSocket.OPEN) {
      socket.addEventListener('open', () => startRecording(), { once: true });
      return;
    }

    socket.send(JSON.stringify({ type: 'control', action: 'start' }));
    setIsRecording(true);
    setTranscript('');
    setAnswer('');

    // Request microphone access and stream audio to the backend.
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });

    mediaRecorder.ondataavailable = async (event) => {
      if (event.data.size > 0 && wsRef.current?.readyState === WebSocket.OPEN) {
        const buffer = await event.data.arrayBuffer();
        const base64Data = arrayBufferToBase64(buffer);
        wsRef.current.send(JSON.stringify({ type: 'audio_chunk', data: base64Data }));
        wsRef.current.send(JSON.stringify({ type: 'transcription_request' }));
      }
    };

    mediaRecorder.start(1000);
    wsRef.current.mediaRecorder = mediaRecorder;

    if (!pollTimer.current) {
      pollTimer.current = setInterval(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({ type: 'poll_transcript' }));
        }
      }, 1000);
    }
  };

  const stopRecording = () => {
    if (wsRef.current?.mediaRecorder && wsRef.current.mediaRecorder.state !== 'inactive') {
      wsRef.current.mediaRecorder.stop();
      wsRef.current.mediaRecorder.stream.getTracks().forEach((track) => track.stop());
      delete wsRef.current.mediaRecorder;
    }
    if (pollTimer.current) {
      clearInterval(pollTimer.current);
      pollTimer.current = null;
    }
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'control', action: 'stop' }));
    }
    setIsRecording(false);
  };

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  return (
    <div className="app">
      <header>
        <h1>Realtime Teleprompter</h1>
        <p className="status">Status: {status}</p>
      </header>
      <main>
        <section className="teleprompter">
          <p className="label">Latest Question</p>
          <div className="scroll-container">
            <p className="teleprompter-text">{transcript || 'Waiting for interviewer...'}</p>
          </div>
        </section>
        <section className="teleprompter answer">
          <p className="label">Suggested Answer</p>
          <div className="scroll-container">
            <p className="teleprompter-text highlight">{answer || 'Your response will appear here.'}</p>
          </div>
        </section>
      </main>
      <footer>
        <button className={isRecording ? 'stop' : 'start'} onClick={toggleRecording}>
          {isRecording ? 'Stop' : 'Start'} Teleprompter
        </button>
      </footer>
    </div>
  );
}

function arrayBufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  bytes.forEach((byte) => {
    binary += String.fromCharCode(byte);
  });
  return btoa(binary);
}
