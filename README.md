# Captioning TTS

This project provides an interactive script to automatically generate image captions and convert them into speech (Text-to-Speech, TTS). Using a combination of computer vision and natural language processing, the script will:

1. Capture or fetch an image (via webcam or URL).
2. Use the BLIP model to generate a descriptive caption for the image.
3. Convert the generated caption into speech using Google Text-to-Speech (gTTS).
4. Play the resulting audio file.

## Features

- **Image acquisition**: Capture images directly from your webcam or download from a provided URL.
- **Image captioning**: Utilizes Salesforce's BLIP (Bootstrapped Language Image Pretraining) model for generating natural language descriptions.
- **Text-to-Speech**: Converts captions to audio using gTTS.
- **Cross-platform audio playback**: Supports Windows, macOS, and Linux for audio output.

## Requirements

- Python 3.7+
- pip (Python package manager)

### Python Libraries

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

#### Main dependencies

- `torch`
- `transformers`
- `Pillow`
- `opencv-python`
- `gTTS`
- `requests`

## Usage

1. **Clone the repository:**

    ```bash
    git clone https://github.com/ayoub4145/captioning_tts.git
    cd captioning_tts
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the application:**

    ```bash
    python app.py
    ```

4. **Follow the prompts:**

    - Choose between:
        - `[1] Webcam` to capture an image from your webcam.
        - `[2] URL` to provide an image URL.

    - The script will generate a caption, synthesize the speech, and play it.

## Notes

- For webcam capture, ensure your webcam is connected and accessible.
- For URL option, provide a direct link to an image (e.g., ending in `.jpg`, `.png`).
- The script will automatically play the generated audio using your system's default player.

## Example

```
Choisissez l'entrée de l'image : [1] Webcam | [2] URL : 2
Entrez l'URL de l'image : https://example.com/cat.png
Génération de la description de l’image...
📝 Description générée : a cat sitting on a sofa.
Génération de la voix...
📢 Lecture du fichier audio...
```

## License

This project is released under the MIT License.

## Acknowledgements

- [Salesforce BLIP](https://github.com/salesforce/BLIP)
- [gTTS (Google Text-to-Speech)](https://pypi.org/project/gTTS/)
- [Transformers by Hugging Face](https://github.com/huggingface/transformers)
