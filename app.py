from PIL import Image
import requests
import torch
from gtts import gTTS
from transformers import BlipProcessor, BlipForConditionalGeneration
import tempfile
import os
import platform
import cv2

# === Étape 1 : Chargement du modèle BLIP ===
print("Chargement du modèle BLIP...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === Étape 2 : Choix de l'image ===
def load_image_from_url(url):
    response = requests.get(url, stream=True, timeout=10)
    return Image.open(response.raw).convert("RGB")

def capture_image_from_webcam():
    cap = cv2.VideoCapture(0)
    print("Appuyez sur [ESPACE] pour capturer une image.")
    while True:
        ret, frame = cap.read()
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == 32:  # Touche espace
            cap.release()
            cv2.destroyAllWindows()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(image)

choice = input("Choisissez l'entrée de l'image : [1] Webcam | [2] URL : ")

if choice == "1":
    image = capture_image_from_webcam()
elif choice == "2":
    url = input("Entrez l'URL de l'image : ")
    try:
        image = load_image_from_url(url)
    except Exception as e:
        print(f"Erreur : {e}")
        exit()
else:
    print("Choix invalide.")
    exit()

# === Étape 3 : Génération de la description ===
print("Génération de la description de l’image...")
inputs = processor(images=image, return_tensors="pt").to(device)
output = model.generate(**inputs)
caption = processor.decode(output[0], skip_special_tokens=True)
print(f"\n📝 Description générée : {caption}")


try:
    print("Génération de la voix...")
    tts = gTTS(text=caption)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        audio_path = fp.name
except Exception as e:
    print(f"Erreur lors de la synthèse vocale : {e}")
    exit()

# === Étape 5 : Lecture audio ===
print(f"📢 Lecture du fichier audio...")

if platform.system() == "Windows":
    os.system(f'start {audio_path}')
elif platform.system() == "Darwin":  # macOS
    os.system(f'afplay {audio_path}')
else:  # Linux
    os.system(f'xdg-open {audio_path}')
