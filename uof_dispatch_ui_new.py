import sys
import queue
import threading
import datetime
import time
import sounddevice as sd
import numpy as np
import whisper
import torch
import scipy.io.wavfile as wav
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QWidget, 
                            QVBoxLayout, QHBoxLayout, QTextEdit, QLabel)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QPalette, QColor, QFont
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


# Constants
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE * 0.5)  # Process in 0.5-second chunks
MIN_PROCESS_INTERVAL = 0.3  # Process every 0.3 seconds
MAX_ACCUMULATED_TIME = 1.0  # Maximum 1 second of audio to accumulate
MIN_AUDIO_LENGTH = SAMPLE_RATE * 0.3  # Minimum 0.3 seconds of audio to process

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
    transcription_correction = pyqtSignal(str)

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
            self.full_audio_buffer = []
            self.transcription_buffer = []
            self.context_window = 5
            self.last_transcription = ""
            self.silence_threshold = 0.01
            self.min_silence_duration = 0.5
            self.last_silence_time = time.time()
            self.accumulated_text = ""
            self.last_chunk_time = time.time()
            self.chunk_timeout = 1.0
            self.audio_buffer = []  # Buffer for accumulating audio
            self.max_buffer_duration = 2.0  # Maximum duration to keep in buffer
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
        current_time = time.time()
        
        # Skip processing if audio is too quiet (unless final)
        if max_amplitude < self.silence_threshold and not is_final:
            return
        
        try:
            # Ensure audio chunk is long enough
            if len(audio_chunk) < MIN_AUDIO_LENGTH and not is_final:
                return
                
            # Ensure audio is 1D
            if len(audio_chunk.shape) > 1:
                audio_chunk = audio_chunk.flatten()
            
            # Add to audio buffer
            self.audio_buffer.append(audio_chunk)
            
            # Keep only recent audio in buffer
            buffer_duration = sum(len(chunk) for chunk in self.audio_buffer) / SAMPLE_RATE
            while buffer_duration > self.max_buffer_duration:
                self.audio_buffer.pop(0)
                buffer_duration = sum(len(chunk) for chunk in self.audio_buffer) / SAMPLE_RATE
            
            # Process the entire buffer for better context
            if len(self.audio_buffer) > 0:
                filename = "temp_audio.wav"
                combined_audio = np.concatenate(self.audio_buffer)
                normalized_audio = combined_audio / np.max(np.abs(combined_audio))
                amplified_audio = normalized_audio * 0.95
                audio_int16 = (amplified_audio * 32767).astype(np.int16)
                wav.write(filename, SAMPLE_RATE, audio_int16)
                
                # Get context from previous transcriptions
                context = self.accumulated_text if self.accumulated_text else ""
                
                # Use more accurate transcription settings with context
                result = self.whisper_model.transcribe(
                    filename,
                    language="en",
                    without_timestamps=True,
                    fp16=False,
                    condition_on_previous_text=True,
                    temperature=0.2,
                    best_of=5,
                    beam_size=5,
                    initial_prompt=context
                )
                
                transcribed_text = result["text"].strip()
                print(f"Raw transcription: {transcribed_text}")
                
                if transcribed_text:
                    # Clean up the transcription
                    transcribed_text = ' '.join(transcribed_text.split())
                    transcribed_text = transcribed_text.replace(" .", ".")
                    
                    # For non-final chunks, update accumulated text
                    if not is_final:
                        # If this is the first chunk, just use the transcribed text
                        if not self.accumulated_text:
                            self.accumulated_text = transcribed_text
                        else:
                            # Find the longest common substring from the end of accumulated_text
                            acc_words = self.accumulated_text.split()
                            new_words = transcribed_text.split()
                            
                            # Try to find overlap
                            max_overlap = 0
                            overlap_pos = 0
                            for i in range(len(acc_words)):
                                for j in range(len(new_words)):
                                    k = 0
                                    while (i + k < len(acc_words) and 
                                           j + k < len(new_words) and 
                                           acc_words[i + k] == new_words[j + k]):
                                        k += 1
                                    if k > max_overlap:
                                        max_overlap = k
                                        overlap_pos = i
                            
                            if max_overlap > 0:
                                # Keep text up to overlap point and add new text from there
                                self.accumulated_text = ' '.join(acc_words[:overlap_pos] + new_words)
                            else:
                                # If no significant overlap found, append with space
                                self.accumulated_text += " " + transcribed_text
                        
                        # Emit the accumulated text
                        self.partial_transcription.emit(self.accumulated_text, True)
                        
                        # Store in transcription buffer for context
                        self.transcription_buffer.append(transcribed_text)
                        
                        # Keep buffer size manageable
                        if len(self.transcription_buffer) > self.context_window * 2:
                            self.transcription_buffer = self.transcription_buffer[-self.context_window:]
                        
                        # Classify force level in real-time
                        try:
                            # Preprocess the text for classification
                            text_to_classify = self.accumulated_text.lower()
                            text_to_classify = ' '.join(text_to_classify.split())
                            
                            # Add context if it's a short phrase
                            if len(text_to_classify.split()) < 5:
                                text_to_classify = "Officer: " + text_to_classify
                            
                            inputs = self.tokenizer(
                                text_to_classify,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=128,
                                add_special_tokens=True
                            )
                            
                            with torch.no_grad():
                                outputs = self.model(**inputs)
                                probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
                                max_prob, prediction = torch.max(probabilities, dim=1)
                            
                            if max_prob.item() < 0.5:
                                prediction = torch.tensor([0])
                                max_prob = torch.tensor([0.5])
                            
                            print(f"Real-time force level: {prediction.item() + 1} with confidence: {max_prob.item():.2%}")
                            self.force_level_ready.emit(prediction.item(), max_prob.item())
                        except Exception as e:
                            print(f"Error in real-time force level classification: {str(e)}")
                    
                    # If we have enough context, try to revise earlier transcriptions
                    if len(self.transcription_buffer) >= self.context_window:
                        self._revise_earlier_transcriptions()
                else:
                    # For final chunks, process all accumulated audio
                    if len(self.full_audio_buffer) > 0:
                        try:
                            # Concatenate all audio and process
                            final_audio = np.concatenate(self.full_audio_buffer)
                            wav.write("final_audio.wav", SAMPLE_RATE, (final_audio * 32767).astype(np.int16))
                            
                            # Use the accumulated text as context for the final transcription
                            final_context = self.accumulated_text if self.accumulated_text else context
                            
                            final_result = self.whisper_model.transcribe(
                                "final_audio.wav",
                                language="en",
                                without_timestamps=True,
                                fp16=False,
                                condition_on_previous_text=True,
                                temperature=0.2,
                                best_of=5,
                                beam_size=5,
                                initial_prompt=final_context
                            )
                            
                            final_text = final_result["text"].strip()
                            if final_text:
                                # Preprocess the text for better classification
                                final_text = final_text.lower()
                                final_text = ' '.join(final_text.split())
                                
                                # Add context if it's a short phrase
                                if len(final_text.split()) < 5:
                                    final_text = "Officer: " + final_text
                                
                                self.transcription_ready.emit(final_text)
                                
                                # Classify force level
                                inputs = self.tokenizer(
                                    final_text,
                                    return_tensors="pt",
                                    padding=True,
                                    truncation=True,
                                    max_length=128,
                                    add_special_tokens=True
                                )
                                
                                with torch.no_grad():
                                    outputs = self.model(**inputs)
                                    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
                                    max_prob, prediction = torch.max(probabilities, dim=1)
                                
                                if max_prob.item() < 0.5:
                                    prediction = torch.tensor([0])
                                    max_prob = torch.tensor([0.5])
                                
                                print(f"Final force level: {prediction.item() + 1} with confidence: {max_prob.item():.2%}")
                                self.force_level_ready.emit(prediction.item(), max_prob.item())
                        except Exception as e:
                            print(f"Error processing final audio: {str(e)}")
                
        except Exception as e:
            error_msg = f"Error processing audio chunk: {str(e)}"
            print(error_msg)
            self.error_occurred.emit(error_msg)

    def _revise_earlier_transcriptions(self):
        """Revise earlier transcriptions using current context."""
        try:
            # Get the last few chunks for context
            recent_context = " ".join(self.transcription_buffer[-self.context_window:])
            
            # Only revise if we have significant changes
            if len(self.transcription_buffer) > self.context_window:
                # Create a context window around all chunks
                context_window = " ".join(self.transcription_buffer)
                
                # Use Whisper to transcribe with full context
                result = self.whisper_model.transcribe(
                    "temp_audio.wav",  # Reuse the current audio file
                    language="en",
                    without_timestamps=True,
                    fp16=False,
                    condition_on_previous_text=True,
                    temperature=0.0,  # Lower temperature for more focused transcription
                    best_of=5,
                    beam_size=5,
                    initial_prompt=context_window
                )
                
                revised_text = result["text"].strip()
                if revised_text and revised_text != self.accumulated_text:
                    # Compare with current accumulated text
                    acc_words = set(self.accumulated_text.lower().split())
                    rev_words = set(revised_text.lower().split())
                    overlap = len(acc_words.intersection(rev_words))
                    total = len(acc_words.union(rev_words))
                    similarity = overlap / total if total > 0 else 0
                    
                    # Only emit correction if the change is significant
                    if similarity < 0.9:  # High threshold to reduce noise
                        self.accumulated_text = revised_text
                        self.transcription_correction.emit(revised_text)
                    
        except Exception as e:
            print(f"Error revising transcriptions: {str(e)}")

    def stop_recording(self):
        if not self.is_recording:
            return
            
        self.is_recording = False
        self.processing_status.emit(True)
        
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
                # Concatenate all audio and process
                final_audio = np.concatenate(self.full_audio_buffer)
                wav.write("final_audio.wav", SAMPLE_RATE, (final_audio * 32767).astype(np.int16))
                
                # Use the accumulated text as context for the final transcription
                final_context = self.accumulated_text if self.accumulated_text else ""
                print(f"Using final context: {final_context}")
                
                # Process the final audio with the full context
                final_result = self.whisper_model.transcribe(
                    "final_audio.wav",
                    language="en",
                    without_timestamps=True,
                    fp16=False,
                    condition_on_previous_text=True,
                    temperature=0.0,  # Use lower temperature for more focused transcription
                    best_of=5,
                    beam_size=5,
                    initial_prompt=final_context
                )
                
                final_text = final_result["text"].strip()
                if final_text:
                    # If we have accumulated text and the final text is very different,
                    # prefer the accumulated text as it was built up in real-time
                    if self.accumulated_text:
                        acc_words = set(self.accumulated_text.lower().split())
                        final_words = set(final_text.lower().split())
                        overlap = len(acc_words.intersection(final_words))
                        total = len(acc_words.union(final_words))
                        similarity = overlap / total if total > 0 else 0
                        
                        if similarity < 0.5:  # If final transcription is very different
                            final_text = self.accumulated_text
                    
                    # Clean up the text
                    final_text = ' '.join(final_text.split())
                    final_text = final_text.replace(" .", ".")
                    
                    # Add context if it's a short phrase
                    if len(final_text.split()) < 5:
                        final_text = "Officer: " + final_text
                    
                    # Emit the final transcription
                    self.transcription_ready.emit(final_text)
                    
                    # Classify force level
                    try:
                        text_to_classify = final_text.lower()
                        text_to_classify = ' '.join(text_to_classify.split())
                        
                        inputs = self.tokenizer(
                            text_to_classify,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=128,
                            add_special_tokens=True
                        )
                        
                        with torch.no_grad():
                            outputs = self.model(**inputs)
                            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
                            max_prob, prediction = torch.max(probabilities, dim=1)
                        
                        if max_prob.item() < 0.5:
                            prediction = torch.tensor([0])
                            max_prob = torch.tensor([0.5])
                        
                        print(f"Final force level: {prediction.item() + 1} with confidence: {max_prob.item():.2%}")
                        self.force_level_ready.emit(prediction.item(), max_prob.item())
                    except Exception as e:
                        print(f"Error in force level classification: {str(e)}")
            except Exception as e:
                self.error_occurred.emit(f"Error processing final audio: {str(e)}")
        
        # Don't clear the accumulated text - we want to keep the live transcription
        self.full_audio_buffer = []
        self.transcription_buffer = []
        self.should_process = False
        self.processing_status.emit(False)
        
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
        self.accumulated_text = ""  # Keep track of all text
        self.last_timestamp = None

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
        
        # Live transcription section
        live_label = QLabel("Live Transcription")
        live_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        live_label.setStyleSheet(f"color: {DARK_THEME['text']};")
        
        self.live_transcription = QTextEdit()
        self.live_transcription.setReadOnly(True)
        self.live_transcription.setMaximumHeight(100)  # Limit height for live transcription
        self.live_transcription.setStyleSheet(f"""
            QTextEdit {{
                background-color: {DARK_THEME['background']};
                color: {DARK_THEME['text']};
                border: 1px solid {DARK_THEME['accent']};
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
            }}
        """)
        
        # Final transcriptions section
        final_label = QLabel("Final Transcriptions")
        final_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        final_label.setStyleSheet(f"color: {DARK_THEME['text']};")
        
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
        
        left_layout.addWidget(live_label)
        left_layout.addWidget(self.live_transcription)
        left_layout.addWidget(final_label)
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
        self.audio_processor.transcription_correction.connect(self.update_transcription_correction)

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
        
        # If we have accumulated text and it's significantly different from the final text,
        # use the accumulated text as it contains the full conversation
        if self.audio_processor.accumulated_text:
            acc_words = set(self.audio_processor.accumulated_text.lower().split())
            final_words = set(text.lower().split())
            overlap = len(acc_words.intersection(final_words))
            total = len(acc_words.union(final_words))
            similarity = overlap / total if total > 0 else 0
            
            if similarity < 0.8 or len(acc_words) > len(final_words):
                text = self.audio_processor.accumulated_text
        
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
            self.transcription_display.clear()  # Clear final transcriptions
            self.live_transcription.clear()  # Clear live transcription
            self.accumulated_text = ""  # Reset accumulated text
            self.last_timestamp = None  # Reset timestamp
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
            # Don't clear the live transcription when stopping
            # The live transcription will be preserved

    def update_partial_transcription(self, text, is_new_sentence):
        if not text.strip():
            return
            
        # Get current timestamp
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        # Always use the full text from the buffer, don't append
        display_text = f"[{timestamp}] {text}"
        
        # Update the display
        self.live_transcription.setPlainText(display_text)
        
        # Scroll to the bottom of live transcription
        self.live_transcription.verticalScrollBar().setValue(
            self.live_transcription.verticalScrollBar().maximum()
        )

    def update_transcription_correction(self, corrected_text):
        # Only show corrections that are significantly different
        current_text = self.transcription_display.toPlainText()
        if not current_text:
            return
            
        # Extract the last correction or transcription
        lines = current_text.split('\n')
        last_text = ""
        for line in reversed(lines):
            if "CORRECTION:" in line or "FINAL TRANSCRIPTION:" in line:
                last_text = line.split(": ", 1)[1] if ": " in line else ""
                break
        
        if last_text:
            # Compare with current correction
            last_words = set(last_text.lower().split())
            new_words = set(corrected_text.lower().split())
            overlap = len(last_words.intersection(new_words))
            total = len(last_words.union(new_words))
            similarity = overlap / total if total > 0 else 0
            
            # Only show correction if it's significantly different
            if similarity < 0.8:
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                correction_text = f"[{timestamp}] CORRECTION: {corrected_text}"
                
                # Add a separator line
                separator = "-" * 50
                self.transcription_display.append(separator)
                self.transcription_display.append(correction_text)
                self.transcription_display.append(separator)
                
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
        