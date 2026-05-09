from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import numpy as np
from PIL import Image
import io
import os

# Initialize FastAPI
app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable - loaded lazily
model = None
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../../models/AgeLens_CNN_Modell.keras')

def load_model_lazy():
    """Load model only when needed"""
    global model
    if model is None:
        try:
            from keras.models import load_model as keras_load_model
            model = keras_load_model(MODEL_PATH)
            print(f"✅ Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise Exception(f"Could not load model: {e}")

# Labels
GENDER_LABELS = {0: "Male", 1: "Female"}
RACE_LABELS = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Other"}

def process_image(image_bytes):
    """
    Process image and prepare for model prediction
    """
    try:
        # Open image and convert to grayscale
        img = Image.open(io.BytesIO(image_bytes)).convert('L')
        
        # Resize to 28x28
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to array and normalize
        img_array = np.array(img).astype('float32') / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)
        
        return img_array
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "message": "AgeLens API is running"
    }

@app.get("/camera-test")
async def camera_test():
    """Camera test page"""
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Kamera Test</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                min-height: 100vh;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
            }
            .container {
                background: white;
                color: #333;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                max-width: 600px;
            }
            h1 { color: #667eea; margin-bottom: 20px; }
            video {
                width: 100%;
                max-width: 400px;
                background: black;
                border-radius: 10px;
                margin: 20px 0;
            }
            button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 12px 30px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 16px;
                margin: 10px 5px;
            }
            button:hover { opacity: 0.9; }
            .status {
                padding: 15px;
                background: #f0f0f0;
                border-radius: 8px;
                margin: 20px 0;
                border-left: 4px solid #667eea;
            }
            .error { color: red; border-left-color: red; }
            .success { color: green; border-left-color: green; }
            .info { color: #667eea; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎥 Kamera Test</h1>
            
            <div id="status" class="status info">
                Klick auf "Kamera Starten" um die Kamera zu testen...
            </div>

            <video id="video" autoplay muted playsinline></video>

            <div>
                <button onclick="startCamera()">🎬 Kamera Starten</button>
                <button onclick="stopCamera()">⏹️ Stop</button>
                <button onclick="checkDevices()">📱 Geräte Prüfen</button>
            </div>

            <div id="devices" class="status" style="display:none;"></div>
        </div>

        <script>
            let stream = null;

            async function startCamera() {
                const status = document.getElementById('status');
                try {
                    status.className = 'status info';
                    status.textContent = 'Requesting camera access...';
                    
                    stream = await navigator.mediaDevices.getUserMedia({
                        video: { facingMode: 'user' },
                        audio: false
                    });
                    
                    const video = document.getElementById('video');
                    video.srcObject = stream;
                    
                    status.className = 'status success';
                    status.textContent = '✅ Kamera erfolgreich gestartet! Du solltest dich sehen.';
                } catch (error) {
                    status.className = 'status error';
                    status.textContent = `❌ Fehler: ${error.message}`;
                    console.error('Camera Error:', error);
                }
            }

            function stopCamera() {
                if (stream) {
                    stream.getTracks().forEach(track => track.stop());
                    document.getElementById('video').srcObject = null;
                    document.getElementById('status').textContent = 'Kamera gestoppt.';
                }
            }

            async function checkDevices() {
                try {
                    const devices = await navigator.mediaDevices.enumerateDevices();
                    const cameras = devices.filter(device => device.kind === 'videoinput');
                    
                    const devicesDiv = document.getElementById('devices');
                    devicesDiv.style.display = 'block';
                    
                    if (cameras.length === 0) {
                        devicesDiv.className = 'status error';
                        devicesDiv.innerHTML = '❌ Keine Kamera gefunden!';
                    } else {
                        devicesDiv.className = 'status success';
                        devicesDiv.innerHTML = `✅ ${cameras.length} Kamera(s) gefunden:<br>` + 
                            cameras.map(cam => `• ${cam.label || 'Unknown Camera'}`).join('<br>');
                    }
                } catch (error) {
                    document.getElementById('devices').className = 'status error';
                    document.getElementById('devices').style.display = 'block';
                    document.getElementById('devices').textContent = `Fehler: ${error.message}`;
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict age, gender, and race from an image
    """
    try:
        # Load model when needed
        load_model_lazy()
        
        if model is None:
            raise HTTPException(status_code=500, detail="Model could not be loaded")
        
        # Read image file
        image_bytes = await file.read()
        
        # Process image
        img_array = process_image(image_bytes)
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        
        # Parse results
        gender_idx = int(np.argmax(predictions[0]))
        gender = GENDER_LABELS.get(gender_idx, "Unknown")
        
        race_idx = int(np.argmax(predictions[1]))
        race = RACE_LABELS.get(race_idx, "Unknown")
        
        age = float(predictions[2][0][0])
        
        # Get confidence scores
        gender_confidence = float(np.max(predictions[0]))
        race_confidence = float(np.max(predictions[1]))
        
        return {
            "status": "success",
            "age": int(age),
            "age_exact": age,
            "gender": gender,
            "gender_confidence": gender_confidence,
            "race": race,
            "race_confidence": race_confidence
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Serve static files (React build)
static_path = os.path.join(os.path.dirname(__file__), 'build')
if os.path.exists(static_path):
    app.mount("/", StaticFiles(directory=static_path, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
