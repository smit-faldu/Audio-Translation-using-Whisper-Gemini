# Audio Translation using Whisper & Gemini

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sKnFmQjS4brkpffvyGIe4F6y1QiV4h33?usp=sharing)

## 📌 Introduction
This project is an **audio-to-text translation system** that transcribes speech from an uploaded `.mp3` or `.wav` file and translates it into the desired language. It utilizes **Whisper Large V3 Turbo** for speech recognition and **Google Gemini** for language translation.

---

## 🚀 Features
- Convert **speech to text** using OpenAI Whisper Large V3 Turbo.
- Translate transcribed text to **multiple languages** using Google Gemini API.
- **User-friendly web interface** built with Flask and TailwindCSS.
- Supports **.mp3 and .wav** audio formats.
- Lightweight and efficient, with **GPU support** for faster processing.

---

## 🎯 Model Used

### 1️⃣ **Whisper Large V3 Turbo** (by OpenAI)
- A powerful **automatic speech recognition (ASR)** model.
- Converts audio speech into text accurately.
- Supports multiple languages for transcription.

### 2️⃣ **Google Gemini API**
- Used for **text translation**.
- Provides fast and accurate language translation.
- Supports a wide range of languages.

---

## 🛠 Installation Guide

### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/smit-faldu/audio-translation.git
cd audio-translation
```

### **2️⃣ Create a Virtual Environment**
```sh
python -m venv venv
source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'
```

### **3️⃣ Install Dependencies**
```sh
pip install -r requirements.txt
```

### **4️⃣ Set Up API Keys** (See next section)

### **5️⃣ Run the Flask Application**
```sh
python main.py
```

---

## 🔑 Setting Up API Keys
This project requires **Google Gemini API Key**.

### **1️⃣ Get Google Gemini API Key**
- Go to [Google AI Studio](https://aistudio.google.com/)
- Sign in with your Google account.
- Generate an API key and **copy it**.
- Create a `.env` file in the project root and add:
  ```env
  GEMINI_API_KEY=your_api_key_here
  ```

---

## 📌 Usage
1. Open the application in your browser.
2. Upload an **audio file** (MP3 or WAV).
3. Select the **target language** for translation.
4. Click **Submit**.
5. The app will display:
   - **Transcribed text** (speech-to-text conversion)
   - **Translated text** in the selected language.

---

## 🏗 Project Structure
```
📂 audio-translation/
├── 📄 main.py               # Flask backend
├── 📄 templates/
│   ├── 📄 index.html       # Frontend UI
├── 📂 uploads/              # Stores uploaded audio files
├── 📄 .env                  # API key (not shared in GitHub)
├── 📄 requirements.txt       # Dependencies list
├── 📄 README.md              # Project documentation
```

---

## 📚 Libraries Used
| Library         | Purpose |
|----------------|------------------------------------------------------------|
| **Flask**      | Web framework for building the API & UI  |
| **Transformers** | Loads OpenAI Whisper model for speech-to-text  |
| **Torch**      | Required for running ML models efficiently |
| **Google Generative AI** | Accesses the Gemini model for translation |
| **dotenv**     | Loads API keys from environment variables |
| **Pyngrok**    | Provides public URL for local hosting |
| **TailwindCSS** | Styles the front-end UI |

---

## ⭐ Contributing
Feel free to **fork** this repository, make changes, and submit a **pull request**!
