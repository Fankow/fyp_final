import React, { useEffect, useState } from "react";
import io from "socket.io-client";
import VideoList from "./VideoList";
import VideoPlayer from "./VideoPlayer";
import ControlPanel from "./ControlPanel"; 
import "./App.css";

// For ngrok, we need to use the right URL
let socketUrl;
if (process.env.REACT_APP_NGROK_URL) {
  // Use the ngrok URL if defined in environment
  socketUrl = process.env.REACT_APP_NGROK_URL;
} else if (window.location.hostname !== "localhost") {
  // If accessed via ngrok or any other domain
  socketUrl = window.location.origin;
} else {
  // When in local development
  socketUrl = "http://localhost:3000";
}

console.log("Connecting to Socket.IO at:", socketUrl);
const socket = io(socketUrl, {
  reconnectionAttempts: 5,
  reconnectionDelay: 1000,
  transports: ["websocket", "polling"],
});

function App() {
  const [frame, setFrame] = useState("");
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [connected, setConnected] = useState(false);
  const [connectionError, setConnectionError] = useState(null);

  useEffect(() => {
    socket.on("connect", () => {
      console.log("Connected to server");
      setConnected(true);
      setConnectionError(null);
    });

    socket.on("connect_error", (err) => {
      console.error("Connection error:", err);
      setConnected(false);
      setConnectionError("Could not connect to server. Check your connection.");
    });

    socket.on("disconnect", () => {
      console.log("Disconnected from server");
      setConnected(false);
    });

    socket.on("frame", (data) => {
      setFrame(`data:image/jpeg;base64,${data}`);
    });

    return () => {
      socket.off("frame");
      socket.off("connect");
      socket.off("connect_error");
      socket.off("disconnect");
    };
  }, []);

  // Function to get correct API URL for videos
  const getApiBaseUrl = () => {
    if (process.env.REACT_APP_NGROK_URL) {
      return process.env.REACT_APP_NGROK_URL;
    } else if (window.location.hostname !== "localhost") {
      return window.location.origin;
    } else {
      return "http://localhost:3000";
    }
  };

  return (
<<<<<<< HEAD
    <div className="App">
<<<<<<< HEAD
      <h1>Raspberry Pi Live Stream with YOLOv11</h1>
      <div className="video-container">
        {frame ? (
          <img src={frame} alt="Live Feed" className="video-feed" />
        ) : (
          <p>Waiting for stream...</p>
        )}
=======
      <header>
        <h1>Raspberry Pi Live Stream with YOLOv11</h1>
      </header>
=======
  <div className="App">
    <header>
      <h1>Raspberry Pi Live Stream with YOLOv11 and Replay System</h1>
    </header>
>>>>>>> ebc544e2 (fixed ui)

    <div className="content">
      {/* Video players in side-by-side row */}
      <div className="video-players-row">
        {/* Live Stream */}
        <div className="video-player-container">
          <h2>
            Live Stream
            {connected ? (
              <span className="status connected"> (Connected)</span>
            ) : (
              <span className="status disconnected"> (Disconnected)</span>
            )}
          </h2>

          {connectionError && (
            <div className="connection-error">{connectionError}</div>
          )}

          {frame ? (
            <img src={frame} alt="Live Feed" className="video-feed" />
          ) : (
            <div className="waiting-stream">
              <p>
                Waiting for stream{connected ? "..." : " (Not connected)"}
              </p>
            </div>
          )}
        </div>

        {/* Recorded Video Player */}
        <div className="control-panel-container">
          <ControlPanel socket={socket} connected={connected} />
        </div>
      </div>

      {/* Controls and Video List Row */}
      <div className="controls-row">
        
        <div className="video-player-container">
          <VideoPlayer videoUrl={selectedVideo} apiBaseUrl={getApiBaseUrl()} />
        </div>
        <div className="video-list-container">
          <VideoList
            onVideoSelect={setSelectedVideo}
            apiBaseUrl={getApiBaseUrl()}
          />
        </div>
>>>>>>> 3bf18ed9 (trying the player and rebuild all the thing)
      </div>
    </div>
  </div>
);
}

export default App;
