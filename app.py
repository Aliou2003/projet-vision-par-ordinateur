import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# --------------------------
# Configuration de la page
# --------------------------
st.set_page_config(
    page_title="Segmentation d'Images",
    page_icon="🖼️",
    layout="wide"
)

# --------------------------
# Titre principal
# --------------------------
st.markdown(
    "<h1 style='text-align: center; color: #4B8BBE;'>🖼️ Segmentation d'Images avec MobileNet</h1>", 
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; color: #333;'>Téléchargez une image et visualisez le masque prédictif avec notre modèle.</p>", 
    unsafe_allow_html=True
)

# --------------------------
# Barre latérale
# --------------------------
st.sidebar.header("📥 Instructions")
st.sidebar.write("""
1. Choisissez une image (jpg, jpeg, png).  
2. Cliquez sur **Segmenter** pour obtenir le masque.  
3. Ajustez la transparence du masque pour voir la superposition en direct.
""")

uploaded_file = st.sidebar.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])
predict_button = st.sidebar.button("Segmenter")

# --------------------------
# Chargement du modèle
# --------------------------
@st.cache_resource
def load_mobilenet_model():
    MODEL_PATH = "model/mobilenet_model.keras"
    return load_model(MODEL_PATH)

model = load_mobilenet_model()

# --------------------------
# Traitement et prédiction
# --------------------------
if uploaded_file is not None and predict_button:
    # Charger et préparer l'image
    image = Image.open(uploaded_file).resize((128, 128))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prédiction
    prediction = model.predict(img_array)
    pred_mask = prediction[0]
    if pred_mask.shape[-1] == 1:
        pred_mask = np.squeeze(pred_mask, axis=-1)
    pred_mask = (pred_mask * 255).astype(np.uint8)

    # Pourcentage de pixels segmentés
    percent_segmented = np.mean(pred_mask > 127) * 100

    # --------------------------
    # Affichage côte à côte
    # --------------------------
    col1, col2, col3 = st.columns(3)

    # Image originale
    with col1:
        st.image(image, caption="Image originale", width=300)
        st.markdown(f"**Taille :** {image.size[0]}x{image.size[1]} px  \n**Format :** {image.format}")

    # Masque prédit
    with col2:
        st.image(pred_mask, caption="Masque prédit", width=300)
        st.markdown(f"**Segmentation détectée : {percent_segmented:.2f}% des pixels**")

    # Superposition avec transparence dynamique
    with col3:
        alpha = st.slider("Transparence du masque", 0.0, 1.0, 0.5, key="alpha")
        overlay = Image.blend(image.convert("RGBA"), Image.fromarray(pred_mask).convert("RGBA"), alpha=alpha)
        st.image(overlay, caption="Superposition", width=300)
        st.markdown(f"**Alpha : {alpha:.2f}**")

# --------------------------
# Footer / Astuce
# --------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>💡 Astuce : utilisez des images de la même taille que celles utilisées pour l'entraînement pour de meilleurs résultats.</p>", 
    unsafe_allow_html=True
)
