FROM node:18-bookworm AS builder

RUN apt-get update && apt-get install -y python3.11 python3-pip && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Abhängigkeiten für den Builder
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/WebApp/package*.json ./src/WebApp/
WORKDIR /app/src/WebApp
RUN npm install
COPY src/WebApp/ .
RUN npm run build


FROM python:3.11-slim-bookworm

ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PYTHONDONTWRITEBYTECODE=1
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