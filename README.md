# Translator-App
# AI-Powered Real-Time Translator

Overview
This project is an AI-powered real-time translator desktop application built with Python and Tkinter. It supports speech recognition, text-to-speech, language detection using a machine learning model, and translation using the Google Translate API (via the `googletrans` library). The app also maintains a history of translations for user reference.

Features
- Real-time text translation between multiple languages.
- Automatic language detection using a custom ML model based on scikit-learn.
- Speech-to-text input using the microphone.
- Text-to-speech output for translated text.
- Translation history saved locally in JSON format.
- User-friendly GUI built with Tkinter.
- Supports over 20 languages including English, French, Spanish, German, Italian, Chinese, Japanese, Hindi, Arabic, and more.
- Auto-detect source language option.
- Auto-speak translated text option.

Installation

Prerequisites
- Python 3.7 or higher
- Required Python packages (install via pip):

```bash
pip install SpeechRecognition pyttsx3 googletrans==4.0.0rc1 pyaudio scikit-learn
```

> Note: Installing `pyaudio` may require additional system dependencies depending on your OS.

Usage

1. Clone or download this repository.
2. Install the required dependencies as shown above.
3. Run the application:

```bash
python app.py
```

4. Use the GUI to input text or start voice input.
5. Select source and target languages or enable auto-detect.
6. Translate text and listen to the spoken translation if desired.
7. View translation history from the History button.

File Descriptions

- `app.py`: Main application code containing the GUI, language detection, translation logic, speech recognition, and text-to-speech functionality.
- `cgi_local.py`: Minimal utility file importing `html.escape`.
- `check_sys_path.py`: Utility script to print the current working directory and Python sys.path for debugging purposes.

Supported Languages
The app supports the following languages (language codes):

- English (en)
- French (fr)
- Spanish (es)
- German (de)
- Italian (it)
- Portuguese (pt)
- Russian (ru)
- Japanese (ja)
- Chinese (zh)
- Korean (ko)
- Hindi (hi)
- Urdu (ur)
- Arabic (ar)
- Bengali (bn)
- Punjabi (pa)
- Malayalam (ml)
- Tamil (ta)
- Telugu (te)
- Kannada (kn)
- Thai (th)
- Vietnamese (vi)
- Indonesian (id)
- Azerbaijani (az)
- Uzbek (uz)

License
This project is open source and free to use.

Acknowledgments
- Uses the `googletrans` library for translation.
- Uses `SpeechRecognition` and `pyaudio` for voice input.
- Uses `pyttsx3` for text-to-speech.
- Language detection powered by a simple scikit-learn ML model.

---

Enjoy seamless real-time translation with this AI-powered desktop app!
