# EdgeAI MusicGen & Stem Splitter 🎵✂️

Een **self-hosted AI-applicatie** voor het genereren van muziek en het splitsen van audiotracks. Dit project draait volledig lokaal ("On Edge") op schoolhardware, geoptimaliseerd voor NVIDIA GPU's en geautomatiseerd via een CI/CD-pipeline met self-hosted runners.

## 🚀 Features

* **Tekst-naar-Muziek:** Genereer unieke tracks op basis van tekstprompts (bijv. *"Techno, 140 BPM, dark atmosphere"*).
* **Stem Separation:** Splits bestaande nummers in 4 sporen: **Drums, Bass, Vocals, Other**.
* **Melody Conditioning:** Upload een melodie die de AI moet volgen.
* **Zero-Downtime Deployment:** Geautomatiseerde updates via GitHub Actions.

## 🏗️ Architectuur

Dit project gebruikt een strikte scheiding tussen Development en Productie om de GPU-resources efficiënt te beheren.

1. **Dev VM (Build Server):**
* Compileert **FFmpeg** handmatig vanuit broncode (voor MP3/AAC codec support).
* Bouwt de Docker image.
* Pusht de image naar de GitHub Container Registry (`ghcr.io`).


2. **Prod VM (Runtime Server):**
* Luistert naar de pipeline via een self-hosted runner.
* Trekt de nieuwste image binnen.
* Herstart de container met behoud van data (via Volumes).



## 🛠️ Technologie Stack

* **Core:** Python 3.9
* **AI Modellen:** Meta AudioCraft (MusicGen), Demucs (Hybrid Transformer)
* **Frontend:** Gradio (Web UI)
* **Containerisatie:** Docker (Multi-stage build)
* **CI/CD:** GitHub Actions (Self-hosted runners)
* **Hardware:** NVIDIA RTX 4000 SFF Ada (CUDA 12.1)

## 📦 Installatie & Lokaal Gebruik

Wil je dit project lokaal draaien? Je hebt een PC nodig met een NVIDIA GPU en Docker geïnstalleerd.

### 1. Clone de repository

```bash
git clone https://github.com/JOUW_NAAM/musicgen-project.git
cd musicgen-project

```

### 2. Bouw de Docker image

Dit kan even duren omdat FFmpeg wordt gecompileerd.

```bash
docker build -t musicgen-app .

```

### 3. Start de Container

Vervang `$(pwd)` door het volledige pad als je op Windows zit.

```bash
docker run --rm -it \
  --gpus all \
  -p 7860:7860 \
  -v "$(pwd)/cache:/root/.cache" \
  -v "$(pwd)/generated:/app/generated" \
  musicgen-app

```

* `--gpus all`: Geeft toegang tot de videokaart.
* `-v .../cache`: Zorgt dat AI-modellen niet steeds opnieuw gedownload worden.

Open je browser en ga naar: `http://localhost:7860`

## ⚙️ CI/CD Pipeline Configuratie

De workflow (`.github/workflows/main.yml`) wordt getriggerd bij elke push naar de `main` branch.

### Jobs:

1. **`build_and_push`** (Draait op Dev VM):
* Bouwt de image.
* Schoont de Docker build cache op om schijfruimte te besparen.


2. **`deploy_prod`** (Draait op Prod VM):
* Wacht tot de build klaar is.
* **Pullt** de nieuwe image.
* **Stopt** de oude container (`musicgen_prod`).
* **Start** de nieuwe container met de juiste volume mounts.



## 🧩 Technische Uitdagingen

### 1. FFmpeg Codecs

De standaard `apt-get install ffmpeg` mist essentiële bibliotheken zoals `libmp3lame`.

* **Oplossing:** Een **Multi-Stage Dockerfile** waarin FFmpeg in de eerste stage handmatig wordt gecompileerd met alle vlaggen (`--enable-libmp3lame`, `--enable-libx264`), en vervolgens naar de eind-image wordt gekopieerd.
