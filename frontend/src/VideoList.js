import React, { useEffect, useState } from "react";
import "./VideoList.css";

function VideoList({ onVideoSelect, apiBaseUrl }) {
  const [videos, setVideos] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);

    // Use the provided API base URL
    const apiUrl = `${apiBaseUrl}/api/videos`;
    console.log("Fetching videos from:", apiUrl);

    fetch(apiUrl)
      .then((res) => {
        if (!res.ok) {
          throw new Error(`HTTP error! Status: ${res.status}`);
        }
        return res.json();
      })
      .then((data) => {
        setVideos(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Error fetching videos:", err);
        setError(err.message);
        setLoading(false);
      });
  }, [apiBaseUrl]);

  // Format file size for display
  const formatFileSize = (bytes) => {
    if (!bytes) return "Unknown";
    const sizes = ["Bytes", "KB", "MB", "GB", "TB"];
    if (bytes === 0) return "0 Bytes";
    const i = parseInt(Math.floor(Math.log(bytes) / Math.log(1024)));
    return Math.round(bytes / Math.pow(1024, i), 2) + " " + sizes[i];
  };

  // Format date for display
  const formatDate = (isoDate) => {
    if (!isoDate) return "Unknown";
    const date = new Date(isoDate);
    return date.toLocaleDateString();
  };

  return (
    <div className="video-list">
      <h2>Available Videos</h2>

      {loading && <p className="status-message">Loading videos...</p>}

      {error && (
        <p className="status-message error">Error loading videos: {error}</p>
      )}

      {!loading && !error && videos.length === 0 && (
        <p className="status-message">No videos found</p>
      )}

      <ul>
        {videos.map((video) => (
          <li key={video.id} onClick={() => onVideoSelect(video)}>
            <div className="video-item-content">
              {video.thumbnail && (
                <img
                  src={video.thumbnail}
                  alt={video.name}
                  className="video-thumbnail"
                />
              )}
              <div className="video-info">
                <div className="video-name">{video.name}</div>
                <div className="video-details">
                  <span>{formatFileSize(video.size)}</span>
                  {video.createdTime && (
                    <span> • {formatDate(video.createdTime)}</span>
                  )}
                </div>
              </div>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default VideoList;
