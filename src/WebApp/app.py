from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import numpy as np
from PIL import Image
import io
import os
from pathlib import Path

# Initialize FastAPI
app = FastAPI(title="AgeLens API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables
age_model = None
gender_model = None

# Model paths
BASE_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
AGE_MODEL_PATH = MODELS_DIR / "best_age_predict_model.keras"
GENDER_MODEL_PATH = MODELS_DIR / "best_gender_predict_model.keras"

def load_models():
    """Load models on startup - using TensorFlow 2.15+ compatible approach"""
    global age_model, gender_model
    try:
        try:
            from tensorflow.keras.models import load_model as keras_load_model
        except ImportError:
            from keras.models import load_model as keras_load_model
        
        import os
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        
        # Load Age Model
        if AGE_MODEL_PATH.exists():
            try:
                age_model = keras_load_model(str(AGE_MODEL_PATH), compile=False, safe_mode=False)
            except TypeError:
                try:
                    age_model = keras_load_model(str(AGE_MODEL_PATH), compile=False)
                except Exception as e2:
                    print(f" Age model error: {str(e2)[:200]}")
            except Exception as e:
                print(f"Error loading age model: {str(e)[:200]}")
        else:
            print(f"Age model not found at {AGE_MODEL_PATH}")
            
        # Load Gender Model
        if GENDER_MODEL_PATH.exists():
            try:
                gender_model = keras_load_model(str(GENDER_MODEL_PATH), compile=False, safe_mode=False)
            except TypeError:
                try:
                    gender_model = keras_load_model(str(GENDER_MODEL_PATH), compile=False)
                except Exception as e2:
                    print(f"Gender model error: {str(e2)[:200]}")
            except Exception as e:
                print(f"Error loading gender model: {str(e)[:200]}")
        else:
            print(f"Gender model not found at {GENDER_MODEL_PATH}")
            
    except Exception as e:
        print(f"Critical error in model loading: {e}")

# Labels
GENDER_LABELS = {0: "Male", 1: "Female"}
RACE_LABELS = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Other"}

def process_image(image_bytes, target_size=(224, 224), grayscale=False):
    """
    Process image and prepare for model prediction
    """
    try:
        # Open image
        img = Image.open(io.BytesIO(image_bytes))
        
        if grayscale:
            # Convert to Grayscale (1 channel)
            img = img.convert('L')
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            img_array = np.array(img, dtype='float32') / 255.0
            # Ensure shape is (224, 224, 1)
            if img_array.ndim == 2:
                img_array = img_array[:, :, np.newaxis]  # Add channel dimension
            # Final shape: (1, 224, 224, 1)
            img_array = img_array[np.newaxis, :, :, :]
        else:
            # Convert to RGB (3 channels)
            img = img.convert('RGB')
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            img_array = np.array(img, dtype='float32') / 255.0
            # Final shape: (1, 224, 224, 3)
            img_array = img_array[np.newaxis, :, :, :]
        
        print(f"Image processed (grayscale={grayscale}): shape={img_array.shape}, dtype={img_array.dtype}, min={img_array.min():.2f}, max={img_array.max():.2f}")
        return img_array
    except Exception as e:
        print(f" Error in process_image: {e}")
        raise Exception(f"Error processing image: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    load_models()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "age_model_loaded": age_model is not None,
        "gender_model_loaded": gender_model is not None,
        "message": "AgeLens API is running"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict age and gender from uploaded image
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    
    if age_model is None or gender_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Please try again.")
    
    try:
        # Read file content
        contents = await file.read()
        
        # Age Model: 224x224, RGB (3 channels)
        img_array_age = process_image(contents, target_size=(224, 224), grayscale=False)
        
        # Gender Model: 48x48, Grayscale (1 channel)
        img_array_gender = process_image(contents, target_size=(48, 48), grayscale=True)
        
        # Make predictions
        print(f"🔮 Predicting age with shape: {img_array_age.shape}")
        age_pred = float(age_model.predict(img_array_age, verbose=0)[0][0])
        
        print(f"🔮 Predicting gender with shape: {img_array_gender.shape}")
        gender_pred_output = gender_model.predict(img_array_gender, verbose=0)[0]
        gender_id = int(np.argmax(gender_pred_output))
        gender_text = GENDER_LABELS.get(gender_id, "Unknown")
        gender_confidence = float(np.max(gender_pred_output))
        
        # Round age to nearest integer
        age_pred_rounded = round(age_pred)
        
        return {
            "success": True,
            "age": age_pred_rounded,
            "age_exact": float(f"{age_pred:.2f}"),
            "gender": gender_text,
            "gender_id": gender_id,
            "confidence": {
                "age": float(f"{age_pred:.2f}"),
                "gender": gender_confidence
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

# Serve static files (React build)
build_dir = Path(__file__).parent / "build"
if build_dir.exists():
    app.mount("/", StaticFiles(directory=str(build_dir), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
