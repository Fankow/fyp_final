const express = require("express");
const http = require("http");
const { Server } = require("socket.io");
const path = require("path");
<<<<<<< HEAD
=======
const { google } = require("googleapis");
const cors = require("cors");
const fs = require("fs");
>>>>>>> 3bf18ed9 (trying the player and rebuild all the thing)

const app = express();
const server = http.createServer(app);

// Improved CORS setup for ngrok
app.use(
  cors({
    origin: "*",
    methods: ["GET", "POST"],
    allowedHeaders: ["Content-Type", "Authorization"],
  })
);

// Socket.IO setup optimized for ngrok and cross-version compatibility
const io = new Server(server, {
  cors: {
<<<<<<< HEAD
    origin: "*", // Allow all origins (update to specific ngrok URL for production)
    methods: ["GET", "POST"],
  },
});

// Serve the React app's static files
app.use(express.static(path.join(__dirname, "../frontend/build")));

// Socket.IO connection handling
=======
    origin: "*",
    methods: ["GET", "POST"],
    transports: ["websocket", "polling"],
    credentials: true,
  },
  allowEIO3: true,
  pingTimeout: 60000,
  pingInterval: 25000, // Add this line
  path: "/socket.io", // Make sure path is explicit
  serveClient: false, // Don't serve client files
});

// Google Drive API setup
const keyFilePath =
  process.env.GOOGLE_SERVICE_ACCOUNT_PATH || "./credentials.json";
let auth;
try {
  if (fs.existsSync(keyFilePath)) {
    auth = new google.auth.GoogleAuth({
      keyFile: keyFilePath,
      scopes: ["https://www.googleapis.com/auth/drive.readonly"],
    });
    console.log("Successfully loaded Google Auth credentials");
  } else {
    console.error("Credentials file not found at:", keyFilePath);
    process.exit(1);
  }
} catch (error) {
  console.error("Error setting up Google Auth:", error);
  process.exit(1);
}

const drive = google.drive({ version: "v3", auth });

// Serve React app static files for production
if (process.env.NODE_ENV === "production") {
  app.use(express.static(path.join(__dirname, "../frontend/build")));
}

// API endpoint to fetch video list from Google Drive
app.get("/api/videos", async (req, res) => {
  const folderId =
    process.env.DRIVE_VIDEOS_FOLDER_ID || "16gNhmALfjDGkLumAcNAPzHIkvSs1OSi7";
  try {
    console.log(`Fetching videos from folder: ${folderId}`);
    const response = await drive.files.list({
      q: `'${folderId}' in parents and mimeType contains 'video/' and trashed=false`,
      fields: "files(id, name, mimeType, thumbnailLink, size, createdTime)",
      orderBy: "name",
    });

    const videos = response.data.files.map((file) => ({
      id: file.id,
      name: file.name,
      mimeType: file.mimeType,
      thumbnail: file.thumbnailLink || null,
      size: file.size,
      createdTime: file.createdTime,
    }));

    console.log(`Found ${videos.length} videos`);
    res.json(videos);
  } catch (error) {
    console.error("Error fetching videos:", error);
    res.status(500).json({ error: error.message });
  }
});

// API endpoint to stream video from Google Drive with range support
app.get("/api/stream/:id", async (req, res) => {
  const fileId = req.params.id;
  try {
    console.log(`Streaming request for file: ${fileId}`);

    // Get file metadata first
    const fileMetadata = await drive.files.get({
      fileId: fileId,
      fields: "name,mimeType,size",
    });

    const fileSize = fileMetadata.data.size;
    const mimeType = fileMetadata.data.mimeType;
    const fileName = fileMetadata.data.name;

    console.log(
      `Streaming file: ${fileName}, Size: ${fileSize}, Type: ${mimeType}`
    );

    // Handle range requests for proper video streaming
    const range = req.headers.range;
    if (range) {
      const parts = range.replace(/bytes=/, "").split("-");
      const start = parseInt(parts[0], 10);
      const end = parts[1] ? parseInt(parts[1], 10) : fileSize - 1;
      const chunkSize = end - start + 1;

      console.log(`Range request: ${start}-${end}/${fileSize}`);

      res.writeHead(206, {
        "Content-Range": `bytes ${start}-${end}/${fileSize}`,
        "Accept-Ranges": "bytes",
        "Content-Length": chunkSize,
        "Content-Type": mimeType,
        "Content-Disposition": `inline; filename="${fileName}"`,
      });

      // Get file with range request
      const response = await drive.files.get(
        {
          fileId: fileId,
          alt: "media",
          headers: {
            Range: `bytes=${start}-${end}`,
          },
        },
        { responseType: "stream" }
      );

      response.data.pipe(res);
    } else {
      // If no range is requested, send the full file
      console.log(`Full file request for: ${fileName}`);

      res.writeHead(200, {
        "Content-Length": fileSize,
        "Content-Type": mimeType,
        "Content-Disposition": `inline; filename="${fileName}"`,
      });

      const response = await drive.files.get(
        { fileId: fileId, alt: "media" },
        { responseType: "stream" }
      );

      response.data.pipe(res);
    }
  } catch (error) {
    console.error("Error streaming video:", error);
    res.status(500).send("Error streaming video");
  }
});

// Socket.IO for live stream
>>>>>>> 3bf18ed9 (trying the player and rebuild all the thing)
io.on("connection", (socket) => {
  console.log("Client connected:", socket.handshake.address);

  // Track user control state
  let hasControl = false; // Initially no one has control

  socket.on("frame", (data) => {
    io.emit("frame", data); // Broadcast frame to all clients
  });

  // Handle PTZ control requests
  socket.on("ptz_control", (direction) => {
    // Forward PTZ control commands to the Raspberry Pi
    io.emit("ptz_command", {
      direction: direction,
      clientId: socket.id,
    });
    console.log(`PTZ command: ${direction} from ${socket.handshake.address}`);
  });

  // Handle manual recording control
  socket.on("recording_control", (action) => {
    // Forward recording control commands to the Raspberry Pi
    io.emit("recording_command", {
      action: action,
      clientId: socket.id,
    });
    console.log(
      `Recording command: ${action} from ${socket.handshake.address}`
    );
  });

  // Handle manual mode toggle
  socket.on("manual_mode", (enabled) => {
    // If requesting manual control
    if (enabled) {
      // Check if someone else already has control
      const controlInfo = io.manualControlInfo || {
        clientId: null,
        timestamp: 0,
      };

      if (controlInfo.clientId && controlInfo.clientId !== socket.id) {
        // Someone else has control, reject request
        socket.emit("manual_mode_response", {
          success: false,
          message: "Another user currently has manual control",
        });
      } else {
        // Grant control to this user
        io.manualControlInfo = {
          clientId: socket.id,
          timestamp: Date.now(),
          address: socket.handshake.address,
        };

        // Inform all clients about the change
        io.emit("control_status_update", {
          status: "manual",
          controlledBy: socket.id,
        });

        // Inform the Raspberry Pi
        io.emit("manual_mode_command", {
          enabled: true,
          clientId: socket.id,
        });

        // Respond to the requesting client
        socket.emit("manual_mode_response", {
          success: true,
          message: "You now have manual control",
        });

        console.log(`Manual mode enabled by ${socket.handshake.address}`);
      }
    } else {
      // Releasing control
      if (io.manualControlInfo && io.manualControlInfo.clientId === socket.id) {
        io.manualControlInfo = { clientId: null, timestamp: 0 };

        // Inform all clients
        io.emit("control_status_update", {
          status: "automatic",
        });

        // Inform the Raspberry Pi
        io.emit("manual_mode_command", {
          enabled: false,
        });

        console.log(`Manual mode disabled by ${socket.handshake.address}`);
      }

      // Always respond with success when disabling
      socket.emit("manual_mode_response", {
        success: true,
        message: "Manual mode disabled",
      });
    }
  });

  // Handle client requests for current control status
  socket.on("get_control_status", () => {
    if (io.manualControlInfo && io.manualControlInfo.clientId) {
      socket.emit("control_status_update", {
        status: "manual",
        controlledBy: io.manualControlInfo.clientId,
        isYou: io.manualControlInfo.clientId === socket.id,
      });
    } else {
      socket.emit("control_status_update", {
        status: "automatic",
      });
    }
  });

  socket.on("disconnect", () => {
    console.log("Client disconnected:", socket.handshake.address);

    // If this client had manual control, release it
    if (io.manualControlInfo && io.manualControlInfo.clientId === socket.id) {
      io.manualControlInfo = { clientId: null, timestamp: 0 };

      // Inform all remaining clients
      io.emit("control_status_update", {
        status: "automatic",
      });

      // Inform the Raspberry Pi
      io.emit("manual_mode_command", {
        enabled: false,
      });

      console.log(`Manual mode auto-disabled due to client disconnect`);
    }
  });
});

<<<<<<< HEAD
// Fallback to serve React app for all routes
app.get("*", (req, res) => {
  res.sendFile(path.join(__dirname, "../frontend/build", "index.html"));
});
=======
// Fallback to serve React app for production
if (process.env.NODE_ENV === "production") {
  app.get("*", (req, res) => {
    res.sendFile(path.join(__dirname, "../frontend/build", "index.html"));
  });
}
>>>>>>> 3bf18ed9 (trying the player and rebuild all the thing)

// Start server
const PORT = process.env.PORT || 3000;
server.listen(PORT, "0.0.0.0", () => {
  console.log(`Server running on http://0.0.0.0:${PORT}`);
  console.log(`For local access: http://localhost:${PORT}`);
});
