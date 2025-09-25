import folium
from folium.plugins import Fullscreen
import branca
from flask import Flask, render_template, jsonify, request
import webbrowser
from pathlib import Path
import json
import logging
from datetime import datetime
import os
from threading import Thread, Event
import time
import pyaudio
import numpy as np
import traceback
import sys

# Audio recording constants
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE * 0.5)  # 0.5 seconds per chunk

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('force_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Disable Flask logging
app.logger.disabled = True
logging.getLogger('werkzeug').disabled = True
os.environ['WERKZEUG_RUN_MAIN'] = 'true'

# Ensure templates directory exists
TEMPLATES_DIR = Path("templates")
TEMPLATES_DIR.mkdir(exist_ok=True)

# Store the latest alert message and transcript entries
latest_alert = {"message": None}
transcript_entries = []

# Add these global variables after the imports
recording_event = Event()
recording_thread = None

class MapInterface:
    def __init__(self, center_lat=33.4255, center_lon=-111.9400):
        """Initialize map centered on Mesa, AZ (or custom coordinates)"""
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.map = None
        self.create_map()
        
    def create_map(self):
        """Create the map with dark theme"""
        # Create base map
        self.map = folium.Map(
            location=[self.center_lat, self.center_lon],
            zoom_start=14,
            tiles="cartodbdark_matter",
            prefer_canvas=True
        )
        
        # Add fullscreen button
        Fullscreen().add_to(self.map)
        
        # Add custom CSS for dark theme and alerts
        custom_css = """
        <style>
            /* ... existing CSS ... */
        </style>
        """
        self.map.get_root().html.add_child(branca.element.Element(custom_css))
        
        # Add HTML template with dark theme
        html_template = """
        <!DOCTYPE html>
        <html>
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Axon Fusus - Real Time Audio Alerts</title>
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
                {{ css }}
            </head>
            <body>
                <div id="topbar">
                    <h1><span>Axon Fusus:</span> Real Time Audio Alerts</h1>
                </div>
                <div id="left-panel">
                    <div id="transcript-header">
                        <h2>Live Transcript</h2>
                    </div>
                    <div id="transcript-content">
                        <!-- Transcript entries will be added here -->
                    </div>
                </div>
                <div id="right-panel">
                    <div id="timeline-header">
                        <h2>Recent Alerts</h2>
                    </div>
                    <div id="timeline-content">
                        <!-- Timeline items will be added here -->
                    </div>
                </div>
                <div id="map">
                    {{ map }}
                </div>
                <div id="alert" class="alert">
                    <strong>⚠️</strong> <span id="alert-message"></span>
                    <span class="close-btn" onclick="dismissAlert()">×</span>
                </div>
                <div id="debug-info"></div>
                {{ script }}
                <div id="recording-status" class="recording-status">
                    <i class="fas fa-microphone"></i>
                    <span>Click the button to start/stop recording</span>
                </div>
                <button id="record-button" class="record-button">Start Recording</button>
            </body>
        </html>
        """
        self.map.get_root().header.add_child(branca.element.Element(html_template))
        
        # Add JavaScript for alert functionality and recording control
        alert_js = """
        <script>
        const MAX_TIMELINE_ITEMS = 50;
        const timelineItems = [];
        let isRecording = false;

        // Add recording button styles
        const style = document.createElement('style');
        style.textContent = `
            .record-button {
                position: fixed;
                bottom: 20px;
                left: 50%;
                transform: translateX(-50%);
                padding: 15px 30px;
                font-size: 16px;
                font-weight: bold;
                color: white;
                background-color: #4CAF50;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                transition: all 0.3s ease;
                z-index: 1000;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            }
            
            .record-button:hover {
                background-color: #45a049;
                transform: translateX(-50%) translateY(-2px);
            }
            
            .record-button.recording {
                background-color: #f44336;
            }
            
            .record-button.recording:hover {
                background-color: #da190b;
            }
        `;
        document.head.appendChild(style);

        // Add recording button functionality
        const recordButton = document.getElementById('record-button');
        recordButton.addEventListener('click', function() {
            isRecording = !isRecording;
            this.textContent = isRecording ? 'Stop Recording' : 'Start Recording';
            this.classList.toggle('recording');
            
            fetch('/toggle_recording', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'recording_started') {
                    document.getElementById('recording-status').innerHTML = 
                        '<i class="fas fa-microphone"></i><span>Recording in progress...</span>';
                } else if (data.status === 'recording_stopped') {
                    document.getElementById('recording-status').innerHTML = 
                        '<i class="fas fa-microphone"></i><span>Click the button to start/stop recording</span>';
                }
            })
            .catch(error => console.error('Error:', error));
        });

        // ... rest of your existing JavaScript code ...
        </script>
        """
        self.map.get_root().html.add_child(branca.element.Element(alert_js))

    # ... rest of your existing MapInterface class methods ...

def start_map_interface(port=5000):
    """Start the Flask web interface."""
    try:
        # Initialize the map
        map_interface = MapInterface()
        # Add the audio marker at the specified coordinates
        map_interface.add_marker(33.64827130544467, -111.89852461418944, 
                              popup="Audio Alert Location",
                              marker_type="audio")
        map_interface.save_map()
        
        # Open the browser in a separate thread
        def open_browser():
            time.sleep(1.5)
            webbrowser.open(f'http://127.0.0.1:{port}')
        
        browser_thread = Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Start server using waitress
        from waitress import serve
        print(f"Starting map interface on port {port}")
        serve(app, host='127.0.0.1', port=port, threads=4)
        
    except Exception as e:
        print(f"Error starting map interface: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Initialize the map
        map_interface = MapInterface()
        # Add the audio marker at the specified coordinates
        map_interface.add_marker(33.64827130544467, -111.89852461418944, 
                              popup="Audio Alert Location",
                              marker_type="audio")
        map_interface.save_map()
        
        # Open the browser in a separate thread
        def open_browser():
            time.sleep(1.5)
            webbrowser.open('http://127.0.0.1:5000')
        
        browser_thread = Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        print("Map interface started. Click the button in the web interface to start/stop recording.")
        print("Press Ctrl+C to exit.")
        
        # Start server using waitress
        from waitress import serve
        serve(app, host='127.0.0.1', port=5000, threads=4)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
        if recording_event.is_set():
            recording_event.clear()
        if recording_thread:
            recording_thread.join()
        sys.exit(0) 