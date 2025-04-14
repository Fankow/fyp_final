CENG 4999 - Final Year Project KWS2401 – Robotic system development

This project is about implementing a rodent detection model for a rodent monitoring system base on YOLOv11 .This repository is the code use for the monitoring system.

The entire project is written with Python as the main application, Node.js and Express.js as the server incorporate with SOCKET.IO, React.js for the frontend.

This project developed for Raspberry Pi. 

![web_ui](https://github.com/user-attachments/assets/c1c0256c-fd54-4576-88b0-a957bf6d123e)

STEPS FOR RUNNING THE WEB ONLINE (NOT LOCALLY)
1.navigate to the backend folder and npm install and  run node server.js
2.ngrok http --url=fyp-web.ngrok.app 3000 (pls use your own in order to expose the server)
3.go to the frontend folder and run npm install and  run npm run build
4.go back to root folder and run the python3 pi_stream.py 
(make sure you have source the env for the virtual environment)

For virtual environment, there is a requirement file for you to do the installation for packages needed for this project.

You will need 4 terminals to run this program.
