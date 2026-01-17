from tensorflow.keras.models import load_model
import os

# Chemin vers le modèle
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../model/mobilenet_model.keras')

def load_mobilenet_model():
    """Charge le modèle MobileNet depuis le dossier model/"""
    model = load_model(MODEL_PATH)
    return model
