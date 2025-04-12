import React, { useState, useEffect } from "react";
import "./ControlPanel.css";

function ControlPanel({ socket, connected }) {
  // State for manual mode toggle
  const [manualMode, setManualMode] = useState(false);
  // State for control status (manual or automatic)
  const [controlStatus, setControlStatus] = useState("automatic");
  // State for whether this user has control
  const [hasControl, setHasControl] = useState(false);
  // State for control error messages
  const [controlMessage, setControlMessage] = useState("");
  // State for recording status
  const [isRecording, setIsRecording] = useState(false);
  // State for whether recording controls are disabled
  const [recordingDisabled, setRecordingDisabled] = useState(false);
  // State for whether PTZ controls are disabled
  const [ptzDisabled, setPtzDisabled] = useState(false);

  // Set up event listeners when component mounts
  useEffect(() => {
    if (socket && connected) {
      // Ask server for current control status
      socket.emit("get_control_status");

      // Listen for control status updates
      socket.on("control_status_update", (data) => {
        // Update control status
        setControlStatus(data.status);
        // Check if we have control
        setHasControl(data.status === "manual" && data.isYou);

        // Show message if someone else has control
        if (data.status === "manual" && !data.isYou) {
          setControlMessage("Another user has manual control");
          setManualMode(false);
        } else if (data.status === "automatic") {
          setControlMessage("");
          setManualMode(false);
        }
      });

      // Listen for response to manual mode toggle
      socket.on("manual_mode_response", (response) => {
        if (!response.success) {
          // Request was denied
          setManualMode(false);
          setControlMessage(response.message);
        } else {
          // Request was successful
          setHasControl(true);
          setControlMessage(response.message);
        }
      });

      // Listen for recording status updates
      socket.on("recording_status", (status) => {
        setIsRecording(status.recording);
        setRecordingDisabled(status.recording && !status.manual);
      });

      // Listen for PTZ status updates
      socket.on("ptz_status", (status) => {
        setPtzDisabled(status.moving && !status.manual);
      });

      // Clean up event listeners when component unmounts
      return () => {
        socket.off("control_status_update");
        socket.off("manual_mode_response");
        socket.off("recording_status");
        socket.off("ptz_status");
      };
    }
  }, [socket, connected]);

  // Toggle manual mode on/off
  const toggleManualMode = () => {
    if (socket && connected) {
      const newState = !manualMode;
      setManualMode(newState);
      socket.emit("manual_mode", newState);
    }
  };

  // Send PTZ control command
  const sendPTZCommand = (direction) => {
    if (socket && connected && hasControl && !ptzDisabled) {
      socket.emit("ptz_control", direction);
    }
  };

  // Toggle recording on/off
  const toggleRecording = () => {
    if (socket && connected && hasControl && !recordingDisabled) {
      const action = isRecording ? "stop" : "start";
      socket.emit("recording_control", action);
      // Update UI immediately (optimistic update)
      setIsRecording(!isRecording);
    }
  };

  return (
    <div className="control-panel">
      <h2>Control Panel</h2>

      {/* Show current control status */}
      <div className="control-status">
        <p>
          System Mode:{" "}
          <span className={`status-${controlStatus}`}>
            {controlStatus.charAt(0).toUpperCase() + controlStatus.slice(1)}
          </span>
        </p>
        {controlMessage && <p className="control-message">{controlMessage}</p>}
      </div>

      {/* Manual mode toggle button */}
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

      {/* PTZ camera controls */}
      <div
        className={`ptz-controls ${
          !hasControl || ptzDisabled ? "disabled" : ""
        }`}
      >
        <h3>PTZ Controls</h3>
        <div className="ptz-grid">
          {/* Up button */}
          <button
            className="ptz-button"
            onClick={() => sendPTZCommand("up")}
            disabled={!hasControl || ptzDisabled}
          >
            ↑
          </button>

          {/* Left button */}
          <button
            className="ptz-button"
            onClick={() => sendPTZCommand("left")}
            disabled={!hasControl || ptzDisabled}
          >
            ←
          </button>

          {/* Right button */}
          <button
            className="ptz-button"
            onClick={() => sendPTZCommand("right")}
            disabled={!hasControl || ptzDisabled}
          >
            →
          </button>

          {/* Down button */}
          <button
            className="ptz-button"
            onClick={() => sendPTZCommand("down")}
            disabled={!hasControl || ptzDisabled}
          >
            ↓
          </button>
        </div>
      </div>

      {/* Recording controls */}
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
