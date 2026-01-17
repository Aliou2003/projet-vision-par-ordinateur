import os
from tensorflow.keras.models import load_model

def test_model_exists():
    model_path = os.path.join("..", "model", "mobilenet_model.keras")
    assert os.path.exists(model_path), "Le modèle n'existe pas"

def test_model_load():
    model_path = os.path.join("..", "model", "mobilenet_model.keras")
    model = load_model(model_path)
    assert model is not None
