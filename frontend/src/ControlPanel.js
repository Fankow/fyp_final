import React, { useState, useEffect } from "react";
import "./ControlPanel.css";

function ControlPanel({ socket, connected }) {
  const [manualMode, setManualMode] = useState(false);
  const [controlStatus, setControlStatus] = useState("automatic");
  const [hasControl, setHasControl] = useState(false);
  const [controlMessage, setControlMessage] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [recordingDisabled, setRecordingDisabled] = useState(false);
  const [ptzDisabled, setPtzDisabled] = useState(false);

  // Request current control status on component mount
  useEffect(() => {
    if (socket && connected) {
      // Request current control status
      socket.emit("get_control_status");

      // Set up event listeners for control status updates
      socket.on("control_status_update", (data) => {
        setControlStatus(data.status);
        setHasControl(data.status === "manual" && data.isYou);

        if (data.status === "manual" && !data.isYou) {
          setControlMessage("Another user has manual control");
          setManualMode(false);
        } else if (data.status === "automatic") {
          setControlMessage("");
          setManualMode(false);
        }
      });

      // Handle response to manual mode toggle
      socket.on("manual_mode_response", (response) => {
        if (!response.success) {
          // If request was denied, update state to reflect this
          setManualMode(false);
          setControlMessage(response.message);
        } else {
          setHasControl(true);
          setControlMessage(response.message);
        }
      });

      // Set up listener for system recording status
      socket.on("recording_status", (status) => {
        setIsRecording(status.recording);
        setRecordingDisabled(status.recording && !status.manual);
      });

      // Set up listener for PTZ status
      socket.on("ptz_status", (status) => {
        setPtzDisabled(status.moving && !status.manual);
      });

      return () => {
        socket.off("control_status_update");
        socket.off("manual_mode_response");
        socket.off("recording_status");
        socket.off("ptz_status");
      };
    }
  }, [socket, connected]);

  // Handle manual mode toggle
  const toggleManualMode = () => {
    if (socket && connected) {
      const newState = !manualMode;
      setManualMode(newState);
      socket.emit("manual_mode", newState);
    }
  };

  // PTZ control functions
  const sendPTZCommand = (direction) => {
    if (socket && connected && hasControl && !ptzDisabled) {
      socket.emit("ptz_control", direction);
    }
  };

  // Recording control functions
  const toggleRecording = () => {
    if (socket && connected && hasControl && !recordingDisabled) {
      const action = isRecording ? "stop" : "start";
      socket.emit("recording_control", action);
      setIsRecording(!isRecording); // Optimistic update
    }
  };

  return (
    <div className="control-panel">
      <h2>Control Panel</h2>

      <div className="control-status">
        <p>
          System Mode:{" "}
          <span className={`status-${controlStatus}`}>
            {controlStatus.charAt(0).toUpperCase() + controlStatus.slice(1)}
          </span>
        </p>
        {controlMessage && <p className="control-message">{controlMessage}</p>}
      </div>

      <div className="toggle-container">
        <span>Manual Control:</span>
        <button
          className={`toggle-button ${manualMode ? "active" : ""}`}
          onClick={toggleManualMode}
          disabled={!connected || (controlStatus === "manual" && !hasControl)}
        >
          {manualMode ? "ON" : "OFF"}
        </button>
      </div>

      <div
        className={`ptz-controls ${
          !hasControl || ptzDisabled ? "disabled" : ""
        }`}
      >
        <h3>PTZ Controls</h3>
        <div className="ptz-grid">
          <button
            className="ptz-button corner"
            onClick={() => sendPTZCommand("up-left")}
            disabled={!hasControl || ptzDisabled}
          >
            ↖
          </button>
          <button
            className="ptz-button"
            onClick={() => sendPTZCommand("up")}
            disabled={!hasControl || ptzDisabled}
          >
            ↑
          </button>
          <button
            className="ptz-button corner"
            onClick={() => sendPTZCommand("up-right")}
            disabled={!hasControl || ptzDisabled}
          >
            ↗
          </button>
          <button
            className="ptz-button"
            onClick={() => sendPTZCommand("left")}
            disabled={!hasControl || ptzDisabled}
          >
            ←
          </button>
          <button
            className="ptz-button center"
            onClick={() => sendPTZCommand("stop")}
            disabled={!hasControl || ptzDisabled}
          >
            ⬤
          </button>
          <button
            className="ptz-button"
            onClick={() => sendPTZCommand("right")}
            disabled={!hasControl || ptzDisabled}
          >
            →
          </button>
          <button
            className="ptz-button corner"
            onClick={() => sendPTZCommand("down-left")}
            disabled={!hasControl || ptzDisabled}
          >
            ↙
          </button>
          <button
            className="ptz-button"
            onClick={() => sendPTZCommand("down")}
            disabled={!hasControl || ptzDisabled}
          >
            ↓
          </button>
          <button
            className="ptz-button corner"
            onClick={() => sendPTZCommand("down-right")}
            disabled={!hasControl || ptzDisabled}
          >
            ↘
          </button>
        </div>
      </div>

      <div
        className={`recording-controls ${
          !hasControl || recordingDisabled ? "disabled" : ""
        }`}
      >
        <h3>Recording Controls</h3>
        <button
          className={`record-button ${isRecording ? "recording" : ""}`}
          onClick={toggleRecording}
          disabled={!hasControl || recordingDisabled}
        >
          {isRecording ? "◼ Stop Recording" : "⬤ Start Recording"}
        </button>
      </div>
    </div>
  );
}

export default ControlPanel;
