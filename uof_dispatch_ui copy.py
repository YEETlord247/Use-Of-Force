import sys
import queue
import threading
import datetime
import time
import sounddevice as sd
import numpy as np
import whisper
import torch
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QWidget, 
                            QVBoxLayout, QHBoxLayout, QTextEdit, QLabel)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QPalette, QColor, QFont
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Constants
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE * 0.5)  # Process in 0.5-second chunks
MIN_PROCESS_INTERVAL = 0.5  # Process every 0.5 seconds
MAX_ACCUMULATED_TIME = 1.0  # Maximum 1 second of audio to accumulate
MIN_AUDIO_LENGTH = SAMPLE_RATE * 0.5  # Minimum 0.5 seconds of audio to process

DARK_THEME = {
    'background': '#1E1E1E',
    'text': '#FFFFFF',
    'button': '#2D2D2D',
    'button_hover': '#3D3D3D',
    'accent': '#007ACC',
    'warning': '#FF4444',
    'success': '#4CAF50'
}

class AudioProcessor(QObject):
    transcription_ready = pyqtSignal(str)
    force_level_ready = pyqtSignal(int, float)
    error_occurred = pyqtSignal(str)
    processing_status = pyqtSignal(bool)
    partial_transcription = pyqtSignal(str, bool)

    def __init__(self):
        super().__init__()
        try:
            print("Initializing AudioProcessor...")
            self.whisper_model = whisper.load_model("base")
            self.tokenizer = DistilBertTokenizer.from_pretrained("./uof_model")
            self.model = DistilBertForSequenceClassification.from_pretrained("./uof_model")
            self.audio_queue = queue.Queue(maxsize=100)
            self.is_recording = False
            self.processing_thread = None
            self.should_process = True
            self.last_transcription = ""
            self.full_audio_buffer = []  # Store all audio during recording
            print("AudioProcessor initialized successfully")
        except Exception as e:
            print(f"Error initializing AudioProcessor: {str(e)}")
            raise

    def start_recording(self):
        print("Starting recording...")
        self.is_recording = True
        self.should_process = True
        self.full_audio_buffer = []  # Clear the buffer when starting new recording
        
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio input status: {status}")
            if self.is_recording:
                try:
                    # Check if we're actually getting audio data with higher sensitivity
                    amplitude = np.max(np.abs(indata))
                    if amplitude > 0:
                        print(f"Audio input detected: max amplitude = {amplitude}")
                    self.audio_queue.put(indata.copy())
                except queue.Full:
                    print("Audio queue is full, dropping frame")
        
        try:
            # List available audio devices
            print("\nAvailable audio devices:")
            devices = sd.query_devices()
            print(devices)
            
            # Find MacBook Pro Microphone
            macbook_mic_idx = None
            for idx, device in enumerate(devices):
                if isinstance(device, dict) and 'name' in device and 'MacBook Pro Microphone' in device['name']:
                    macbook_mic_idx = idx
                    break
            
            if macbook_mic_idx is None:
                print("MacBook Pro Microphone not found, using default input device")
                device_info = sd.query_devices(kind='input')
            else:
                print(f"Using MacBook Pro Microphone (index {macbook_mic_idx})")
                device_info = devices[macbook_mic_idx]
            
            print(f"\nUsing input device: {device_info}")
            
            # Create the input stream with adjusted settings
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=audio_callback,
                blocksize=CHUNK_SIZE,
                device=macbook_mic_idx,  # Use MacBook microphone
                latency='low'  # Use low latency mode
            )
            self.stream.start()
            print("Audio stream started successfully")
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self._process_audio_thread)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            print("Processing thread started")
            
        except Exception as e:
            error_msg = f"Failed to start recording: {str(e)}"
            print(error_msg)
            self.error_occurred.emit(error_msg)
            self.is_recording = False

    def process_audio_chunk(self, audio_chunk, is_final=False):
        max_amplitude = np.max(np.abs(audio_chunk))
        if max_amplitude < 0.001:
            return
        
        try:
            # Ensure audio chunk is long enough
            if len(audio_chunk) < MIN_AUDIO_LENGTH and not is_final:
                return
                
            # Ensure audio is 1D
            if len(audio_chunk.shape) > 1:
                audio_chunk = audio_chunk.flatten()
            
            import scipy.io.wavfile as wav
            filename = "temp_audio.wav"
            normalized_audio = audio_chunk / np.max(np.abs(audio_chunk))
            amplified_audio = normalized_audio * 0.95
            audio_int16 = (amplified_audio * 32767).astype(np.int16)
            wav.write(filename, SAMPLE_RATE, audio_int16)
            
            # Use more accurate transcription settings
            result = self.whisper_model.transcribe(
                filename,
                language="en",
                without_timestamps=True,
                fp16=False,
                condition_on_previous_text=False,  # Don't condition on previous for better accuracy
                temperature=0.0
            )
            
            transcribed_text = result["text"].strip()
            print(f"Raw transcription: {transcribed_text}")  # Debug output
            
            if transcribed_text:
                # Clean up the transcription
                transcribed_text = ' '.join(transcribed_text.split())
                transcribed_text = transcribed_text.replace(" .", ".")
                
                if not is_final:
                    # For live updates, emit the current transcription
                    self.partial_transcription.emit(transcribed_text, True)
                else:
                    # For final chunks, process all accumulated audio
                    if len(self.full_audio_buffer) > 0:
                        try:
                            # Concatenate all audio and process
                            final_audio = np.concatenate(self.full_audio_buffer)
                            wav.write("final_audio.wav", SAMPLE_RATE, (final_audio * 32767).astype(np.int16))
                            
                            final_result = self.whisper_model.transcribe(
                                "final_audio.wav",
                                language="en",
                                without_timestamps=True,
                                fp16=False,
                                condition_on_previous_text=False,
                                temperature=0.0
                            )
                            
                            final_text = final_result["text"].strip()
                            if final_text:
                                self.transcription_ready.emit(final_text)
                                
                                # Classify force level
                                inputs = self.tokenizer(final_text, return_tensors="pt", 
                                                      padding=True, truncation=True, max_length=128)
                                outputs = self.model(**inputs)
                                probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
                                max_prob, prediction = torch.max(probabilities, dim=1)
                                
                                print(f"Predicted force level: {prediction.item() + 1} with confidence: {max_prob.item():.2%}")
                                self.force_level_ready.emit(prediction.item(), max_prob.item())
                        except Exception as e:
                            print(f"Error processing final audio: {str(e)}")
                
        except Exception as e:
            error_msg = f"Error processing audio chunk: {str(e)}"
            print(error_msg)
            self.error_occurred.emit(error_msg)

    def stop_recording(self):
        if not self.is_recording:
            return
            
        self.is_recording = False
        self.processing_status.emit(True)  # Indicate processing started
        
        # Clean up stream
        if hasattr(self, 'stream'):
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                self.error_occurred.emit(f"Error stopping stream: {str(e)}")
        
        # Process any remaining audio in the queue
        while not self.audio_queue.empty():
            try:
                chunk = self.audio_queue.get_nowait()
                if len(chunk.shape) > 1:
                    chunk = chunk.flatten()
                self.full_audio_buffer.append(chunk)
            except queue.Empty:
                break
        
        # Process the complete audio buffer
        if self.full_audio_buffer:
            try:
                final_audio = np.concatenate(self.full_audio_buffer)
                self.process_audio_chunk(final_audio, is_final=True)
            except Exception as e:
                self.error_occurred.emit(f"Error processing final audio: {str(e)}")
        
        self.full_audio_buffer = []  # Clear the buffer
        self.should_process = False
        self.processing_status.emit(False)  # Indicate processing completed
        
        # Wait for processing thread to complete
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)

    def _process_audio_thread(self):
        accumulated_chunks = []
        last_process_time = time.time()
        accumulated_duration = 0
        print("Audio processing thread started")
        
        while self.should_process:
            current_time = time.time()
            
            # Get any new audio chunks
            while not self.audio_queue.empty():
                try:
                    chunk = self.audio_queue.get_nowait()
                    if len(chunk.shape) > 1:
                        chunk = chunk.flatten()
                    accumulated_chunks.append(chunk)
                    self.full_audio_buffer.append(chunk.copy())  # Store in full buffer
                    accumulated_duration += len(chunk) / SAMPLE_RATE
                except queue.Empty:
                    break
            
            # Process if we have enough audio duration or enough time has passed
            if accumulated_chunks and (
                current_time - last_process_time >= MIN_PROCESS_INTERVAL or
                accumulated_duration >= MAX_ACCUMULATED_TIME
            ):
                try:
                    audio = np.concatenate(accumulated_chunks)
                    self.process_audio_chunk(audio)
                    
                    # Clear accumulated chunks after processing
                    accumulated_chunks = []
                    accumulated_duration = 0
                    last_process_time = current_time
                except Exception as e:
                    print(f"Error processing audio: {str(e)}")
                    accumulated_chunks = []
                    accumulated_duration = 0
            
            time.sleep(0.01)  # Shorter sleep for more responsive updates

class DispatchUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.audio_processor = AudioProcessor()
        self.setup_ui()
        self.setup_connections()
        self.current_line = ""
        self.current_timestamp = None
        self.current_text = ""
        self.chunk_counter = 0  # Add counter for chunks

    def setup_ui(self):
        self.setWindowTitle("Use of Force Dispatch Monitor")
        self.setMinimumSize(1000, 600)
        
        # Set up the main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # Left panel - Transcription
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        transcription_label = QLabel("Live Transcription")
        transcription_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        transcription_label.setStyleSheet(f"color: {DARK_THEME['text']};")
        
        self.transcription_display = QTextEdit()
        self.transcription_display.setReadOnly(True)
        self.transcription_display.setStyleSheet(f"""
            QTextEdit {{
                background-color: {DARK_THEME['background']};
                color: {DARK_THEME['text']};
                border: 1px solid {DARK_THEME['accent']};
                border-radius: 5px;
                padding: 10px;
            }}
        """)
        
        left_layout.addWidget(transcription_label)
        left_layout.addWidget(self.transcription_display)
        
        # Right panel - Controls and Force Level
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Record button
        self.record_button = QPushButton("Start Recording")
        self.record_button.setMinimumHeight(50)
        self.record_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {DARK_THEME['button']};
                color: {DARK_THEME['text']};
                border: none;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {DARK_THEME['button_hover']};
            }}
        """)
        
        # Force level display
        self.force_level_display = QLabel("Force Level: --")
        self.force_level_display.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.force_level_display.setStyleSheet(f"color: {DARK_THEME['text']};")
        self.force_level_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Confidence display
        self.confidence_display = QLabel("Confidence: --")
        self.confidence_display.setFont(QFont("Arial", 12))
        self.confidence_display.setStyleSheet(f"color: {DARK_THEME['text']};")
        self.confidence_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Status indicator
        self.status_indicator = QLabel("Ready")
        self.status_indicator.setFont(QFont("Arial", 12))
        self.status_indicator.setStyleSheet(f"color: {DARK_THEME['success']};")
        self.status_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Error display
        self.error_display = QLabel("")
        self.error_display.setFont(QFont("Arial", 10))
        self.error_display.setStyleSheet(f"color: {DARK_THEME['warning']};")
        self.error_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.error_display.setWordWrap(True)
        
        right_layout.addWidget(self.record_button)
        right_layout.addWidget(self.force_level_display)
        right_layout.addWidget(self.confidence_display)
        right_layout.addWidget(self.status_indicator)
        right_layout.addWidget(self.error_display)
        right_layout.addStretch()
        
        # Add panels to main layout
        layout.addWidget(left_panel, stretch=2)
        layout.addWidget(right_panel, stretch=1)
        
        # Set dark theme for window
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {DARK_THEME['background']};
            }}
            QWidget {{
                background-color: {DARK_THEME['background']};
            }}
        """)

    def setup_connections(self):
        self.record_button.clicked.connect(self.toggle_recording)
        self.audio_processor.transcription_ready.connect(self.update_transcription)
        self.audio_processor.partial_transcription.connect(self.update_partial_transcription)
        self.audio_processor.force_level_ready.connect(self.update_force_level)
        self.audio_processor.error_occurred.connect(self.show_error)
        self.audio_processor.processing_status.connect(self.update_processing_status)

    def show_error(self, error_message):
        self.error_display.setText(error_message)
        QTimer.singleShot(5000, lambda: self.error_display.setText(""))  # Clear after 5 seconds

    def update_processing_status(self, is_processing):
        if is_processing:
            self.status_indicator.setText("Processing...")
            self.status_indicator.setStyleSheet(f"color: {DARK_THEME['warning']};")
        else:
            self.status_indicator.setText("Ready")
            self.status_indicator.setStyleSheet(f"color: {DARK_THEME['success']};")

    def update_transcription(self, text):
        if self.transcription_display.toPlainText():
            self.transcription_display.insertPlainText("\n\n")
        
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        final_text = f"[{timestamp}] FINAL TRANSCRIPTION: {text}"
        
        # Add a separator line
        separator = "-" * 50
        self.transcription_display.append(separator)
        self.transcription_display.append(final_text)
        self.transcription_display.append(separator)
        
        # Scroll to the bottom
        self.transcription_display.verticalScrollBar().setValue(
            self.transcription_display.verticalScrollBar().maximum()
        )

    def update_force_level(self, level, confidence):
        level_colors = {
            0: "#4CAF50",  # Green - Level 1
            1: "#8BC34A",  # Light Green - Level 2
            2: "#FFC107",  # Yellow - Level 3
            3: "#FF9800",  # Orange - Level 4
            4: "#F44336"   # Red - Level 5
        }
        
        # Ensure level is between 0-4 (corresponding to levels 1-5)
        level = max(0, min(4, level))
        
        self.force_level_display.setText(f"Force Level: {level + 1}")
        self.force_level_display.setStyleSheet(f"color: {level_colors.get(level, DARK_THEME['text'])};")
        self.confidence_display.setText(f"Confidence: {confidence:.2%}")
        
        if level >= 3:  # High force level alert
            self.force_level_display.setStyleSheet(f"""
                color: {level_colors.get(level, DARK_THEME['text'])};
                font-weight: bold;
                text-decoration: underline;
            """)

    def toggle_recording(self):
        if not self.audio_processor.is_recording:
            self.audio_processor.start_recording()
            self.record_button.setText("Stop Recording")
            self.transcription_display.clear()  # Clear display when starting new recording
            self.chunk_counter = 0  # Reset chunk counter
            self.record_button.setStyleSheet(f"""
                QPushButton {{
                    background-color: {DARK_THEME['warning']};
                    color: {DARK_THEME['text']};
                    border: none;
                    border-radius: 5px;
                    padding: 10px;
                    font-size: 14px;
                    font-weight: bold;
                }}
            
                QPushButton:hover {{
                    background-color: #FF6666;
                }}
            """)
            self.status_indicator.setText("Recording...")
            self.status_indicator.setStyleSheet(f"color: {DARK_THEME['warning']};")
        else:
            self.audio_processor.stop_recording()
            self.record_button.setText("Start Recording")
            self.record_button.setStyleSheet(f"""
                QPushButton {{
                    background-color: {DARK_THEME['button']};
                    color: {DARK_THEME['text']};
                    border: none;
                    border-radius: 5px;
                    padding: 10px;
                    font-size: 14px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: {DARK_THEME['button_hover']};
                }}
            """)

    def update_partial_transcription(self, text, is_new_sentence):
        if not text.strip():
            return
            
        cursor = self.transcription_display.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.transcription_display.setTextCursor(cursor)
        
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        new_chunk = f"[{timestamp}] {text}"  # Removed "Chunk X:" label
        
        # Get current text and add new chunk
        current_text = self.transcription_display.toPlainText()
        if current_text:
            updated_text = current_text + "\n" + new_chunk
        else:
            updated_text = new_chunk
            
        self.transcription_display.setPlainText(updated_text)
        self.chunk_counter += 1  # Still increment counter for internal tracking
        
        # Scroll to the bottom
        self.transcription_display.verticalScrollBar().setValue(
            self.transcription_display.verticalScrollBar().maximum()
        )

def main():
    app = QApplication(sys.argv)
    window = DispatchUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
        