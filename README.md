USE-OF-FORCE DETECTION PIPELINE
------------------------------------------------------------
An AI-driven audio and NLP pipeline that detects and classifies 
use-of-force incidents from real-world police dispatch recordings.
Developed at Axon Enterprises by Utkarsh Rai.

------------------------------------------------------------
OVERVIEW
------------------------------------------------------------
The Use-of-Force Detection Pipeline processes live or recorded 
dispatch audio to automatically detect potential use-of-force events. 
The system combines automatic speech recognition (ASR) using 
OpenAI’s Whisper model with fine-tuned BERT Transformers to 
identify high-risk incidents based on semantic content and 
linguistic tone.

The output is displayed on an interactive dashboard that maps 
incident frequency and severity in real time, allowing for 
data-driven review and analytics.

------------------------------------------------------------
SYSTEM ARCHITECTURE
------------------------------------------------------------
Audio Input (Dispatch Radio / Recording)
   ↓
Whisper Speech-to-Text Engine
   ↓
Text Preprocessing + Tokenization
   ↓
Fine-Tuned BERT Classifier
   ↓
Classification Output (Incident Type + Confidence)
   ↓
Flask + Plotly Dashboard Visualization

Core pipeline components:
1. Audio transcription using Whisper (fine-tuned on noisy dispatch audio)
2. Sentence-level classification using BERT for intent and escalation detection
3. Visualization via geospatial dashboard (Flask + Plotly)
4. RESTful API endpoints for streaming classification and retraining

------------------------------------------------------------
KEY FEATURES
------------------------------------------------------------
- 95%+ accuracy on balanced dataset of dispatch recordings
- End-to-end ASR + NLP integration
- Whisper fine-tuned for high-noise, low-quality radio inputs
- GPU-accelerated transcription and classification
- Flask-based dashboard for interactive exploration
- Supports both batch uploads and real-time audio streaming
- Automatic data logging and retraining workflow

------------------------------------------------------------
TECH STACK
------------------------------------------------------------
Languages: Python 3.10+
ASR: OpenAI Whisper
NLP: PyTorch, Transformers (BERT, DistilBERT)
Visualization: Flask, Plotly, Pandas
Data Tools: NumPy, Scikit-learn, TorchScript
Deployment: REST API / Flask Server
Hardware: NVIDIA GPU (CUDA Enabled)

------------------------------------------------------------
REPOSITORY STRUCTURE
------------------------------------------------------------
use_of_force_detection/
 ├── fine_tune_whisper.py        (Whisper fine-tuning + inference)
 ├── evaluate_model.py           (Classification evaluation and metrics)
 ├── final_balanced_dataset.csv  (Labeled dataset)
 ├── map_interface.py            (Flask + Plotly dashboard)
 ├── monitor_training.py         (Model performance logger)
 ├── logs/                       (Classification and latency logs)
 └── README.txt

------------------------------------------------------------
SETUP INSTRUCTIONS
------------------------------------------------------------
1. Prerequisites
   - Python >= 3.8
   - CUDA-compatible GPU
   - Access to OpenAI Whisper weights
   - Hugging Face Transformers installed

2. Installation
   git clone <repo-url>
   cd use_of_force_detection
   pip install -r requirements.txt

3. Running the Pipeline
   python fine_tune_whisper.py
   python evaluate_model.py
   python map_interface.py

------------------------------------------------------------
USAGE
------------------------------------------------------------
1. Load dispatch audio files (.wav, .mp3, or .m4a).
2. The system transcribes the audio using Whisper.
3. The BERT classifier analyzes the transcript to detect 
   use-of-force language or escalation.
4. Detected incidents are displayed on the live dashboard 
   with coordinates, timestamps, and classification confidence.

------------------------------------------------------------
PERFORMANCE BENCHMARKS
------------------------------------------------------------
Transcription Accuracy (Whisper):        ~93%
Classification Accuracy (BERT):          ~95%
Processing Latency (GPU):                ~500 ms per segment
Audio Dataset Size:                      10,000+ samples
Supported Audio Formats:                 WAV, MP3, M4A
Dashboard Update Interval:               <1 second

------------------------------------------------------------
DESIGN HIGHLIGHTS
------------------------------------------------------------
- Optimized batch transcription with parallel Whisper threads
- Fine-tuned BERT classifier using domain-specific data augmentation
- Dynamic retraining loop based on new dispatch logs
- RESTful API endpoints for automated ingestion
- Real-time error handling and confidence visualization
- Modular code design for easy integration with Axon products

------------------------------------------------------------
EXAMPLE OUTPUT
------------------------------------------------------------
Audio Input:
   "Unit 23 to dispatch, suspect is resisting. Officer requesting backup."

Output:
   Transcription: "Unit 23 to dispatch, suspect is resisting. Officer requesting backup."
   Classification: USE-OF-FORCE (Confidence: 0.97)
   Timestamp: 00:02:17
   Geolocation: 33.452° N, 112.074° W

------------------------------------------------------------
FUTURE WORK
------------------------------------------------------------
- Integrate multilingual transcription support
- Deploy Whisper inference on edge devices for live patrol audio
- Add multimodal context (audio + video fusion)
- Extend classification taxonomy (de-escalation, verbal threat, etc.)
- Integrate with Evidence.com for direct archival and playback

------------------------------------------------------------
AUTHOR
------------------------------------------------------------
Utkarsh Rai
R&D Intern — Axon Enterprises
Email: rai.utkarsh2007@gmail.com
LinkedIn: linkedin.com/in/utkarsh-rai-7249611b6
