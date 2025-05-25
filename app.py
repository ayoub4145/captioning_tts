import streamlit as st
from PIL import Image
import requests
import tempfile
import torch
from gtts import gTTS
from transformers import BlipProcessor, BlipForConditionalGeneration

# Charger BLIP
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return processor, model, device

processor, model, device = load_model()

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
            response = requests.get(url, stream=True)
            image = Image.open(response.raw).convert("RGB")
            st.image(image, caption="Image chargée", use_column_width=True)
        except Exception as e:
            st.error(f"Erreur lors du chargement de l'image : {e}")

# Choix de la langue
lang_map = {
    "Français": "fr",
    "Anglais": "en",
    "Arabe": "ar"
}
lang_choice = st.selectbox("Choisissez la langue de la synthèse vocale :", list(lang_map.keys()))
lang_code = lang_map[lang_choice]

# Si une image est chargée
if image:
    st.image(image, caption="Image sélectionnée", use_column_width=True)

    # Bouton pour générer
    if st.button("🧠 Générer la description et lire à haute voix"):
        with st.spinner("Génération de la description..."):
            inputs = processor(image, return_tensors="pt").to(device)
            output = model.generate(**inputs)
            caption = processor.decode(output[0], skip_special_tokens=True)

        st.success("✅ Description :")
        st.markdown(f"**📝 {caption}**")

        # Synthèse vocale
        with st.spinner(f"Synthèse vocale ({lang_choice})..."):
            tts = gTTS(text=caption, lang=lang_code)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts.save(fp.name)
                st.audio(fp.name, format="audio/mp3")
