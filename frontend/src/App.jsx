import React, { useMemo, useState } from "react";
import { useEffect } from "react";

const API_BASE = "http://127.0.0.1:8000";
const CHECKPOINTS = [
    "Uploading video",
    "Preparing audio with FFmpeg",
    "Running speech-to-text model",
    "Formatting subtitle segments",
    "Subtitles ready",
];

const MODEL_INFO = {
    "qwen3-asr-0.6b": {
        name: "Qwen3-ASR-0.6B",
        note: "Best for Mandarin/English mix quality; slower on CPU and faster with GPU.",
    },
    tiny: {
        name: "Whisper Tiny",
        note: "Fastest, lowest accuracy. Good for quick drafts.",
    },
    base: {
        name: "Whisper Base",
        note: "Balanced speed and quality for short clips.",
    },
    small: {
        name: "Whisper Small",
        note: "Default sweet spot on CPU.",
    },
    medium: {
        name: "Whisper Medium",
        note: "Better quality, slower processing.",
    },
    "large-v3": {
        name: "Whisper Large v3",
        note: "Best default for Mandarin + English mixed audio, with higher runtime cost.",
    },
};

function App() {
    const [videoFile, setVideoFile] = useState(null);
    const [videoUrl, setVideoUrl] = useState("");
    const [modelSize, setModelSize] = useState("large-v3");
    const [language, setLanguage] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [editedSrt, setEditedSrt] = useState("");
    const [error, setError] = useState("");
    const [checkpointIndex, setCheckpointIndex] = useState(-1);
    const [elapsedSeconds, setElapsedSeconds] = useState(0);

    const subtitleUrl = useMemo(() => {
        if (!editedSrt) return "";
        const blob = new Blob([editedSrt], { type: "application/x-subrip" });
        return URL.createObjectURL(blob);
    }, [editedSrt]);

    useEffect(() => {
        if (!isLoading) return undefined;

        setCheckpointIndex(0);
        setElapsedSeconds(0);

        const elapsedTimer = window.setInterval(() => {
            setElapsedSeconds((prev) => prev + 1);
        }, 1000);

        // Backend currently returns once completed, so these are user-facing stage hints.
        const phaseTimer = window.setInterval(() => {
            setCheckpointIndex((prev) => Math.min(prev + 1, 3));
        }, 3500);

        return () => {
            window.clearInterval(elapsedTimer);
            window.clearInterval(phaseTimer);
        };
    }, [isLoading]);

    useEffect(() => {
        return () => {
            if (videoUrl) URL.revokeObjectURL(videoUrl);
            if (subtitleUrl) URL.revokeObjectURL(subtitleUrl);
        };
    }, [videoUrl, subtitleUrl]);

    const onPickVideo = (event) => {
        const file = event.target.files?.[0];
        if (!file) return;

        setVideoFile(file);
        setResult(null);
        setEditedSrt("");
        setError("");

        if (videoUrl) {
            URL.revokeObjectURL(videoUrl);
        }
        setVideoUrl(URL.createObjectURL(file));
    };

    const onSubmit = async (event) => {
        event.preventDefault();
        if (!videoFile) {
            setError("Please choose a video first.");
            return;
        }

        const formData = new FormData();
        formData.append("file", videoFile);
        formData.append("model_size", modelSize);
        if (language.trim()) formData.append("language", language.trim());

        setIsLoading(true);
        setCheckpointIndex(0);
        setError("");

        try {
            const response = await fetch(`${API_BASE}/api/transcribe`, {
                method: "POST",
                body: formData,
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.detail || "Transcription failed.");
            }
            setResult(data);
            setEditedSrt(data.srt || "");
            setCheckpointIndex(4);
        } catch (err) {
            setError(err.message || "Unexpected error.");
            setCheckpointIndex(-1);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="page">
            <main className="shell">
                <h1>Local Subtitle Tool</h1>
                <p className="hint">Upload a video, transcribe speech locally, and review the generated SRT before download.</p>

                <section className="workspace">
                    <section className="leftPane">
                        <form className="controls" onSubmit={onSubmit}>
                            <label>
                                Video File
                                <input type="file" accept="video/*" onChange={onPickVideo} />
                            </label>

                            <label>
                                Model Size
                                <select value={modelSize} onChange={(e) => setModelSize(e.target.value)}>
                                    <option value="qwen3-asr-0.6b">Qwen3-ASR-0.6B (Mandarin/English, GPU recommended)</option>
                                    <option value="large-v3">Whisper Large v3 (recommended for Mandarin + English mix)</option>
                                    <option value="medium">Whisper Medium</option>
                                    <option value="small">Whisper Small</option>
                                    <option value="base">Whisper Base</option>
                                    <option value="tiny">Whisper Tiny (fastest)</option>
                                </select>
                            </label>
                            <p className="modelHint">
                                <strong>{MODEL_INFO[modelSize].name}</strong>: {MODEL_INFO[modelSize].note}
                            </p>

                            <label>
                                Language (optional, e.g. en, id, ja)
                                <input
                                    type="text"
                                    placeholder="auto-detect if empty"
                                    value={language}
                                    onChange={(e) => setLanguage(e.target.value)}
                                />
                            </label>

                            <button type="submit" disabled={isLoading || !videoFile}>
                                {isLoading ? "Transcribing..." : "Generate Subtitles"}
                            </button>
                        </form>

                        {(isLoading || checkpointIndex >= 0) && (
                            <section className="progressCard">
                                <div className="progressHead">
                                    <h2>Progress</h2>
                                    {isLoading && <span>{elapsedSeconds}s</span>}
                                </div>
                                <ol className="checkpointList">
                                    {CHECKPOINTS.map((label, index) => {
                                        const status = index < checkpointIndex ? "done" : index === checkpointIndex ? "current" : "pending";
                                        return (
                                            <li key={label} className={`checkpoint ${status}`}>
                                                <span className="dot" aria-hidden="true" />
                                                <span>{label}</span>
                                            </li>
                                        );
                                    })}
                                </ol>
                            </section>
                        )}

                        {error && <p className="error">{error}</p>}

                        {result && (
                            <section className="resultCard">
                                <p>
                                    Detected language: <strong>{result.language}</strong> ({(result.language_probability * 100).toFixed(1)}%)
                                </p>
                                <p>Segments: {result.segments.length}</p>
                                <a href={subtitleUrl} download="subtitles.edited.srt" className="download">
                                    Download Edited SRT
                                </a>
                            </section>
                        )}

                        {result && (
                            <section className="srtCard">
                                <h2>SRT Editor</h2>
                                <p className="muted">Edit text and timestamps directly. Video subtitle preview updates automatically.</p>
                                <textarea
                                    className="srtEditor"
                                    value={editedSrt}
                                    onChange={(event) => setEditedSrt(event.target.value)}
                                    spellCheck={false}
                                />
                                <div className="editorActions">
                                    <button type="button" className="ghostBtn" onClick={() => setEditedSrt(result.srt)}>
                                        Reset To Generated
                                    </button>
                                </div>
                            </section>
                        )}
                    </section>

                    <section className="rightPane">
                        {videoUrl ? (
                            <video controls src={videoUrl} className="video">
                                {subtitleUrl && <track kind="subtitles" src={subtitleUrl} srcLang="en" label="Generated subtitles" default />}
                            </video>
                        ) : (
                            <div className="placeholder">Video preview will appear here.</div>
                        )}
                    </section>
                </section>
            </main>
        </div>
    );
}

export default App;
