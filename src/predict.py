import numpy as np
from keras.models import load_model
from PIL import Image

# 1. Modell laden
# Ersetze 'AgeLens_CNN_Modell.keras' durch deinen tatsächlichen Dateinamen
model = load_model('AgeLens_CNN_Modell.keras')

# Hilfslisten für die Textausgabe
GENDER_LABELS = {0: "Mann", 1: "Frau"}
RACE_LABELS = {0: "Weiß", 1: "Schwarz", 2: "Asiatisch", 3: "Indisch", 4: "Andere"}

def predict_face(image_path):
    try:
        # 2. Bild einlesen und vorbereiten (genau wie beim Training)
        img = Image.open(image_path).convert('L') # Graustufen
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Umwandeln in Array, Normalisieren und Shape anpassen (1, 28, 28, 1)
        img_array = np.array(img).astype('float32') / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        # 3. Vorhersage machen
        # Das Modell gibt eine Liste zurück: [gender_preds, race_preds, age_preds]
        predictions = model.predict(img_array, verbose=0)

        # 4. Ergebnisse auswerten
        # Gender (Klassifikation -> Index mit höchstem Wert)
        gender_idx = np.argmax(predictions[0])
        gender_text = GENDER_LABELS[gender_idx]

        # Race (Klassifikation -> Index mit höchstem Wert)
        race_idx = np.argmax(predictions[1])
        race_text = RACE_LABELS[race_idx]

        # Age (Regression -> der direkte Wert)
        age_val = float(predictions[2][0][0])

        # 5. Ausgabe in der Konsole
        print("-" * 30)
        print(f"ERGEBNIS FÜR: {image_path}")
        print(f"Geschlecht: {gender_text}")
        print(f"Ethnie:     {race_text}")
        print(f"Alter:      ca. {int(age_val)} Jahre")
        print("-" * 30)

    except Exception as e:
        print(f"Fehler beim Verarbeiten des Bildes: {e}")

# --- BEISPIEL-AUFRUF ---
# Gib hier den Pfad zu einem Bild ein, das das Modell noch nicht kennt
mein_bild = "pfad/zu/deinem/testbild.jpg"
predict_face(mein_bild)