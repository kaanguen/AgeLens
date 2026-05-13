# --- STAGE 1: Build React Frontend ---
FROM node:18-bookworm AS builder

# Python 3.11 für mögliche Build-Scripts nachinstallieren
RUN apt-get update && apt-get install -y python3.11 python3-pip && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PEP 668 Fix für Debian Bookworm (erlaubt pip install global)
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Abhängigkeiten für den Builder
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# React Build Prozess
COPY src/WebApp/package*.json ./src/WebApp/
WORKDIR /app/src/WebApp
RUN npm install
COPY src/WebApp/ .
RUN npm run build


# --- STAGE 2: Runtime (Python Stage) ---
FROM python:3.11-slim-bookworm

# PEP 668 Fix auch in der Runtime Stage
ENV PIP_BREAK_SYSTEM_PACKAGES=1
# Verhindert, dass Python .pyc Dateien schreibt
ENV PYTHONDONTWRITEBYTECODE=1
# Sorgt für sofortige Log-Ausgabe (kein Buffering)
ENV PYTHONUNBUFFERED=1

# System-Libraries für OpenCV (wichtig für Image-Processing im Container)
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Requirements kopieren und installieren
# TIPP: Ändere tensorflow in der requirements.txt auf deine lokale Version (z.B. 2.16.1)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Den restlichen App-Code kopieren
COPY . .

# Den fertigen React-Build aus Stage 1 übernehmen
COPY --from=builder /app/src/WebApp/build ./src/WebApp/build

# Port für FastAPI
EXPOSE 8000

# Start-Befehl
CMD ["python", "src/WebApp/run_server.py"]