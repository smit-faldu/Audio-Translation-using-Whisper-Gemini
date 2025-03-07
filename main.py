
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import base64
import os
from google import genai
from google.genai import types
from flask import Flask, request, render_template
from pyngrok import ngrok, conf
from dotenv import load_dotenv


GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

app = Flask(__name__)

# Create an 'uploads' directory to store audio files
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)




processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

languages = {
    "af": "Afrikaans", "sq": "Albanian", "am": "Amharic", "ar": "Arabic",
    "hy": "Armenian", "as": "Assamese", "az": "Azerbaijani", "bn": "Bangla",
    "eu": "Basque", "be": "Belarusian", "bs": "Bosnian", "bg": "Bulgarian",
    "ca": "Catalan", "zh-CN": "Chinese (Simplified)", "zh-TW": "Chinese (Traditional)",
    "hr": "Croatian", "cs": "Czech", "da": "Danish", "fa-AF": "Dari", "nl": "Dutch",
    "en": "English", "et": "Estonian", "fil": "Filipino", "fi": "Finnish", "fr": "French",
    "gl": "Galician", "ka": "Georgian", "de": "German", "el": "Greek", "gu": "Gujarati",
    "ht": "Haitian Creole", "ha": "Hausa", "he": "Hebrew", "hi": "Hindi", "hu": "Hungarian",
    "is": "Icelandic", "ig": "Igbo", "id": "Indonesian", "ga": "Irish", "it": "Italian",
    "ja": "Japanese", "jv": "Javanese", "kn": "Kannada", "kk": "Kazakh", "km": "Khmer",
    "ko": "Korean", "ku": "Kurdish", "ky": "Kyrgyz", "lo": "Lao", "lv": "Latvian",
    "lt": "Lithuanian", "lb": "Luxembourgish", "mk": "Macedonian", "ms": "Malay",
    "ml": "Malayalam", "mt": "Maltese", "mi": "Maori", "mr": "Marathi", "mn": "Mongolian",
    "my": "Myanmar (Burmese)", "ne": "Nepali", "no": "Norwegian", "or": "Odia (Oriya)",
    "ps": "Pashto", "fa": "Persian (Farsi)", "pl": "Polish", "pt": "Portuguese", "pa": "Punjabi",
    "ro": "Romanian", "ru": "Russian", "sm": "Samoan", "gd": "Scots Gaelic", "sr": "Serbian",
    "sd": "Sindhi", "si": "Sinhala (Sinhalese)", "sk": "Slovak", "sl": "Slovenian", "so": "Somali",
    "es": "Spanish", "sw": "Swahili", "sv": "Swedish", "tg": "Tajik", "ta": "Tamil", "te": "Telugu",
    "th": "Thai", "tr": "Turkish", "tk": "Turkmen", "uk": "Ukrainian", "ur": "Urdu", "ug": "Uyghur",
    "uz": "Uzbek", "vi": "Vietnamese", "cy": "Welsh", "xh": "Xhosa", "yi": "Yiddish", "yo": "Yoruba",
    "zu": "Zulu"
}

def generate_translation(text, target_language):
    client = genai.Client(
        api_key=GEMINI_API_KEY,  # Replace with your valid API key
    )

    model = "gemini-2.0-flash-exp"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=f"Translate the following text to {target_language}. Only return the translated sentence without any explanations or additional text.\n\n{text}"
                ),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        response_mime_type="text/plain",
    )

    translated_text = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        translated_text += chunk.text
    return translated_text



@app.route("/", methods=["GET", "POST"])
def home():
    transcribed_text = None
    file_name = None
    selected_language = None
    translated_text = None

    if request.method == "POST":
        file = request.files["file"]
        selected_language = request.form["language"]

        if file and file.filename.endswith((".mp3", ".wav")):
            file_name = file.filename
            file_path = f"uploads/{file.filename}"
            os.makedirs("uploads", exist_ok=True)
            file.save(file_path)

            # Convert speech to text
            output = pipe(file_path)
            transcribed_text = output["text"]

            translated_text = generate_translation(transcribed_text, languages[selected_language])

    return render_template("index.html", transcribed_text = transcribed_text , translated_text = translated_text , file_name=file_name, selected_language=selected_language, languages=languages)

if __name__ == "__main__":
    port = 5000
    public_url = ngrok.connect(port).public_url
    print(f"Public URL: {public_url}")
    app.run(port=port)