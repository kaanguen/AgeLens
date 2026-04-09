# AgeLens - Altersvorhersage mit Deep Learning

Ein Deep Learning-Projekt zur Vorhersage des menschlichen Alters aus Gesichtsbildern mithilfe des UTKFace-Datensatzes.



## 🛠️ Installation

```bash
pip install -r requirements.txt
```

## 🚀 Verwendung

### 1. Datensatz herunterladen
- Datensatz von [Kaggle](https://www.kaggle.com/datasets/jangedoo/utkface-new) herunterladen
- Bilder in `data/raw/` entpacken

### 2. Modell trainieren
```bash
python src/train.py
```

### 3. Web-App starten
```bash
streamlit run app.py
```

### 4. Vorhersage für einzelnes Bild
```python
from src.predict import predict_age
age = predict_age('path/to/image.jpg')
print(f"Geschätztes Alter: {age}")
```

## 📦 Dependencies

Siehe `requirements.txt`:
- TensorFlow - Deep Learning
- OpenCV - Bildverarbeitung
- NumPy & Pandas - Datenmanipulation
- Streamlit/Gradio - Web-Interface
- Scikit-learn - ML-Utilities

## 📝 Notes

- ⚠️ Datensatz wird NICHT auf GitHub hochgeladen (siehe .gitignore)
- 💾 Trainierte Modelle sollten versioniert werden (v1, v2, etc.)
- 📊 Nutze `exploration.ipynb` zur Datensatzanalyse