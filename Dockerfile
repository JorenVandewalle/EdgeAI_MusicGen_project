# --- STAGE 1: Bouwen van FFmpeg ---
FROM python:3.9-slim-bookworm AS builder

# Installeer tools om te bouwen
RUN apt-get update && \
    apt-get install -y build-essential pkg-config wget yasm nasm libtool autoconf automake \
    libx264-dev libx265-dev libvpx-dev libmp3lame-dev && \
    apt-get clean

# Download en compileer FFmpeg
WORKDIR /tmp
RUN wget https://ffmpeg.org/releases/ffmpeg-5.1.tar.bz2 && \
    tar xjf ffmpeg-5.1.tar.bz2 && \
    cd ffmpeg-5.1 && \
    ./configure --prefix=/usr/local --enable-shared --disable-doc --enable-gpl \
                --enable-version3 --enable-nonfree --disable-x86asm --disable-inline-asm \
                --enable-libx264 --enable-libx265 --enable-libvpx --enable-libmp3lame && \
    make -j$(nproc) && \
    make install

# --- STAGE 2: De Echte Image (Runtime) ---
FROM python:3.9-slim-bookworm

# Installeren build-essential en pkg-config
RUN apt-get update && \
    apt-get install -y build-essential pkg-config \
    libx264-dev libx265-dev libvpx-dev libmp3lame-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Kopieer de gebouwde FFmpeg van de vorige stap
COPY --from=builder /usr/local /usr/local

# Update de bibliotheek-links
RUN ldconfig

# 1. Installeer PyTorch
RUN python -m pip install --no-cache-dir torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Installeer PyAV
RUN python -m pip install --no-cache-dir av==11.0.0

# 3. Installeer AudioCraft, Demucs en de rest
RUN python -m pip install --no-cache-dir audiocraft "gradio==3.50.2" xformers "numpy<2.0" "transformers==4.37.2" demucs

# Maak de werkmap aan
WORKDIR /app

# Kopieer jouw code naar de container
COPY . .

# Open de poort
EXPOSE 7860

# Start het commando
CMD ["python", "full_app.py", "--listen", "0.0.0.0", "--server_port", "7860"]