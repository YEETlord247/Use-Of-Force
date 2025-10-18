# Real-Time Translation System


A bilingual real-time communication system enabling seamless officer–civilian interaction via instant audio translation. Built at Axon as a proof-of-concept reviewed by the CEO and later transitioned into production development.


## Overview
The system translates spoken audio between two languages in real time using Azure Speech-to-Text and Google Translate APIs. The platform employs a push-to-talk duplex interface for low-latency bidirectional communication.


## Architecture
1. **Audio Capture** – Captured via PyAudio stream
2. **Speech-to-Text** – Azure Cognitive Services converts audio to text
3. **Translation** – Google Translate API converts text to target language
4. **Text-to-Speech** – Output spoken through secondary audio channel
5. **Networking** – UDP sockets handle duplex streaming between devices


## Key Features
- Sub-300 ms round-trip translation latency  
- Fully asynchronous duplex communication channel  
- Supports multiple language pairs with auto-detection  
- Robust error handling and fallback mechanisms for unstable networks  
- Modular pipeline supporting both CLI and GUI-based interfaces  


## Tech Stack
Python, Azure Speech SDK, Google Translate API, PyAudio, Socket Programming, AsyncIO


## Setup
