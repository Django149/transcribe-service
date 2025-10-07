# Use PyTorch with CUDA support as base image for GPU acceleration
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# Set working directory
WORKDIR /app

# Configure LD_LIBRARY_PATH for CUDA libraries
ENV LD_LIBRARY_PATH="/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib"

# Install system dependencies
# - ffmpeg: for audio/video processing
# - libmagic1: for python-magic file type detection
# - gcc, g++, make: for building Python packages
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Fix torchvision compatibility issue for pyannote.audio
RUN pip install --no-cache-dir --upgrade torchvision torchaudio

# Install speechbrain explicitly for speaker recognition
RUN pip install --no-cache-dir speechbrain

# Accept HuggingFace token as build argument for gated models
ARG HF_TOKEN

# Pre-download all AI models at build time to avoid runtime downloads
# Whisper models for transcription (public, no auth needed)
RUN python3 -c 'import faster_whisper; m = faster_whisper.WhisperModel("ivrit-ai/whisper-large-v3-turbo-ct2")'
RUN python3 -c "import os; import pyannote.audio; p = pyannote.audio.Pipeline.from_pretrained('ivrit-ai/pyannote-speaker-diarization-3.1')"
RUN python3 -c 'from speechbrain.inference.speaker import EncoderClassifier; EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")'

# Copy application files
COPY . .

# Run the application
CMD ["python", "app.py", "--dev"]
