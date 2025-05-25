import streamlit as st
from PIL import Image
import requests
import tempfile
import torch
from gtts import gTTS
from transformers import BlipProcessor, BlipForConditionalGeneration

# Chargement du modèle et du processeur
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Utilisation de GPU si disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Titre de l'application
st.title("📸 BLIP avec Webcam, URL & Synthèse vocale 🔊")
st.markdown("Décrivez automatiquement une image avec BLIP et écoutez la description en plusieurs langues.")

# Choix de l'entrée d'image
option = st.radio("Choisissez une méthode :", ["📷 Webcam", "🌐 URL de l'image"])
image = None

if option == "📷 Webcam":
    uploaded_frame = st.camera_input("Prenez une photo avec votre webcam")
    if uploaded_frame:
        image = Image.open(uploaded_frame).convert("RGB")

elif option == "🌐 URL de l'image":
    url = st.text_input("Entrez l'URL d'une image :")
    if url:
        try:
            response = requests.get(url, stream=True, timeout=10)
            image = Image.open(response.raw).convert("RGB")
            st.image(image, caption="Image chargée", use_column_width=True)
        except Exception as e:
            st.error(f"Erreur lors du chargement de l'image : {e}")

# Choix de la langue pour la synthèse vocale
lang_map = {
    "Français": "fr",
    "Anglais": "en",
    "Arabe": "ar"
}
lang_choice = st.selectbox("Choisissez la langue de la synthèse vocale :", list(lang_map.keys()))
lang_code = lang_map[lang_choice]

# Si une image est présente
if image:
    st.image(image, caption="Image sélectionnée", use_column_width=True)

    if st.button("🧠 Générer la description et lire à haute voix"):
        with st.spinner("Génération de la description..."):
            inputs = processor(images=image, return_tensors="pt").to(device)
            output = model.generate(**inputs)
            caption = processor.decode(output[0], skip_special_tokens=True)

        st.success("✅ Description :")
        st.markdown(f"**📝 {caption}**")

        # Synthèse vocale
        with st.spinner(f"Synthèse vocale ({lang_choice})..."):
            try:
                tts = gTTS(text=caption, lang=lang_code)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                    tts.save(fp.name)
                    st.audio(fp.name, format="audio/mp3")
            except Exception as e:
                st.error(f"Erreur de synthèse vocale : {e}")
