# STAGE 1: Bouwer (Compileren)
FROM python:3.9-bookworm AS builder

# Installeer alleen wat nodig is om te bouwen
RUN apt-get update && \
    apt-get install -y build-essential pkg-config wget yasm nasm libtool autoconf automake \
    libx264-dev libx265-dev libvpx-dev libmp3lame-dev && \
    apt-get clean

# Compileer FFmpeg
WORKDIR /tmp
RUN wget https://ffmpeg.org/releases/ffmpeg-5.1.tar.bz2 && \
    tar xjf ffmpeg-5.1.tar.bz2 && \
    cd ffmpeg-5.1 && \
    ./configure --prefix=/usr/local --enable-shared --disable-doc --enable-gpl \
                --enable-version3 --enable-nonfree --disable-x86asm --disable-inline-asm \
                --enable-libx264 --enable-libx265 --enable-libvpx --enable-libmp3lame && \
    make -j$(nproc) && \
    make install

# ---------------------------------------------------------

# STAGE 2: Eind Image (Alleen runtime)
FROM python:3.9-bookworm

# Installeer runtime libraries (geen compilers meer nodig!)
RUN apt-get update && \
    apt-get install -y libx264-dev libx265-dev libvpx-dev libmp3lame-dev && \
    apt-get remove -y libavcodec-dev libavformat-dev libavutil-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Kopieer de gecompileerde FFmpeg uit de 'builder' stage
COPY --from=builder /usr/local /usr/local

# Update library links
RUN ldconfig

# Environment
ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
ENV LD_LIBRARY_PATH=/usr/local/lib
ENV CFLAGS="-I/usr/local/include"
ENV LDFLAGS="-L/usr/local/lib"

# Python Packages (Nu hebben we veel meer ruimte!)
RUN python -m pip install --no-cache-dir torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN python -m pip install --no-cache-dir av==11.0.0
RUN python -m pip install --no-cache-dir audiocraft "gradio==3.50.2" xformers "numpy<2.0" "transformers==4.37.2"

WORKDIR /app
EXPOSE 7860
CMD ["python", "-u", "full_app.py", "--listen", "0.0.0.0", "--server_port", "7860"]