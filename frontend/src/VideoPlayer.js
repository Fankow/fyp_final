import React, { useState, useEffect, useRef } from "react";
import "./VideoPlayer.css";

function VideoPlayer({ videoUrl, apiBaseUrl }) {
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  // eslint-disable-next-line
  const [isPlaying, setIsPlaying] = useState(false);
  const videoRef = useRef(null);

  useEffect(() => {
    // Reset states when video changes
    if (videoUrl) {
      setLoading(true);
      setError(null);
      setIsPlaying(false);
    }
  }, [videoUrl]);

  const handleLoadStart = () => {
    setLoading(true);
    setIsPlaying(false);
  };

  const handleCanPlay = () => {
    setLoading(false);
  };

  const handlePlay = () => {
    setIsPlaying(true);
  };

  const handlePause = () => {
    setIsPlaying(false);
  };

  const handleError = (e) => {
    console.error("Video playback error:", e);
    setLoading(false);
    setIsPlaying(false);
    setError("Failed to play video in browser. You can download and play locally instead.");
  };

  // Get the correct video streaming URL
  const getVideoUrl = (video) => {
    if (!video) return null;
    return `${apiBaseUrl}/api/stream/${video.id}`;
  };
  
  // Get the download URL
  const getDownloadUrl = () => {
    if (!videoUrl) return null;
    return `${apiBaseUrl}/api/stream/${videoUrl.id}`;
  };

  return (
    <div className="video-player">
      <h2>Video Player</h2>

      {!videoUrl && (
        <div className="empty-player">
          <p>Select a video to play</p>
        </div>
      )}

      {videoUrl && (
        <div className="player-container">
          <h3 className="video-title">{videoUrl.name}</h3>

          {loading && <div className="loading-indicator">Loading video...</div>}

          {error && <div className="error-message">{error}</div>}

          <video
            ref={videoRef}
            controls
            src={getVideoUrl(videoUrl)}
            width="100%"
            onLoadStart={handleLoadStart}
            onCanPlay={handleCanPlay}
            onPlay={handlePlay}
            onPause={handlePause}
            onError={handleError}
            playsInline
          >
            Your browser does not support the video tag.
          </video>
          
          {/* Always show download button */}
          <div className="download-container">
            <a 
              href={getDownloadUrl()} 
              download={videoUrl.name}
              className="download-button"
              target="_blank"
              rel="noopener noreferrer"
            >
              Download Video
            </a>
          </div>

          <div className="video-info-box">
            {videoUrl.mimeType && (
              <div className="video-detail">
                <span className="detail-label">Format:</span>
                <span className="detail-value">
                  {videoUrl.mimeType.split("/")[1].toUpperCase()}
                </span>
              </div>
            )}
            {videoUrl.size && (
              <div className="video-detail">
                <span className="detail-label">Size:</span>
                <span className="detail-value">
                  {formatFileSize(videoUrl.size)}
                </span>
              </div>
            )}
            {videoUrl.createdTime && (
              <div className="video-detail">
                <span className="detail-label">Created:</span>
                <span className="detail-value">
                  {new Date(videoUrl.createdTime).toLocaleString()}
                </span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// Format file size for display
function formatFileSize(bytes) {
  if (!bytes) return "Unknown";
  const sizes = ["Bytes", "KB", "MB", "GB", "TB"];
  if (bytes === 0) return "0 Bytes";
  const i = parseInt(Math.floor(Math.log(bytes) / Math.log(1024)));
  return Math.round(bytes / Math.pow(1024, i), 2) + " " + sizes[i];
}

export default VideoPlayer;
