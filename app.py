import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import queue
import time
import json
import os
from datetime import datetime
import speech_recognition as sr
import pyttsx3
import googletrans
import pyaudio
import wave
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle

# Define supported languages dictionary
LANGUAGES = {
    "en": "english",
    "fr": "french",
    "es": "spanish",
    "de": "german",
    "it": "italian",
    "pt": "portuguese",
    "ru": "russian",
    "ja": "japanese",
    "zh": "chinese",
    "ko": "korean",
    "hi": "hindi",
    "ur": "urdu",
    "ar": "arabic",
    "bn": "bengali",
    "pa": "punjabi",
    "ml": "malayalam",
    "ta": "tamil",
    "te": "telugu",
    "kn": "kannada",
    "th": "thai",
    "vi": "vietnamese",
    "id": "indonesian",
    "az": "azerbaijani",
    "uz": "uzbek"
}

class LanguageDetector:
    """Language detection using scikit-learn"""
    
    def __init__(self):
        self.model = None
        self.setup_model()
    
    def setup_model(self):
        """Setup a simple language detection model"""
        training_texts = [
            ("Hello how are you today", "en"),         # English
    ("Bonjour comment allez-vous", "fr"),      # French
    ("Hola como estas hoy", "es"),             # Spanish
    ("Hallo wie geht es dir", "de"),           # German
    ("Ciao come stai oggi", "it"),             # Italian
    ("Olá como você está", "pt"),              # Portuguese
    ("Привет как дела", "ru"),                 # Russian
    ("こんにちは元気ですか", "ja"),            # Japanese
    ("你好吗今天", "zh"),                        # Chinese (Simplified)
    ("안녕하세요 어떻게 지내세요", "ko"),          # Korean
    ("नमस्ते आप कैसे हैं", "hi"),               # Hindi
    ("ہیلو آپ کیسے ہیں", "ur"),                  # Urdu
    ("مرحبًا كيف حالك اليوم", "ar"),             # Arabic
    ("হ্যালো আপনি কেমন আছেন", "bn"),           # Bengali
    ("ਸਤ ਸ੍ਰੀ ਅਕਾਲ ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ", "pa"),       # Punjabi
    ("ഹലോ നിങ്ങൾ ഇന്ന് എങ്ങിനെയാണ്", "ml"),     # Malayalam
    ("வணக்கம் இன்று எப்படி இருக்கிறீர்கள்", "ta"), # Tamil
    ("హలో మీరు ఈరోజు ఎలా ఉన్నారు", "te"),         # Telugu
    ("ನಮಸ್ಕಾರ ನೀವು ಇವತ್ತು ಹೇಗಿದ್ದೀರಿ", "kn"),     # Kannada
    ("สวัสดีวันนี้คุณเป็นอย่างไรบ้าง", "th"),     # Thai
    ("Xin chào hôm nay bạn thế nào", "vi"),     # Vietnamese
    ("Halo apa kabar hari ini", "id"),          # Indonesian
    ("Salam, bugünkü halınız necədir?", "az"),  # Azerbaijani
    ("Salom, bugun qalaysiz?", "uz"),           # Uzbek
        ]
        
        texts, labels = zip(*training_texts)
        
        # Create pipeline with TF-IDF and Naive Bayes
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(1, 3))),
            ('clf', MultinomialNB())
        ])
        
        self.model.fit(texts, labels)
    
    def detect(self, text):
        """Detect language of given text"""
        if not text.strip():
            return "en"  # Default to English
        
        try:
            prediction = self.model.predict([text])[0]
            return prediction
        except:
            return "en"  # Fallback to English

class TranslationHistory:
    """Manage translation history"""
    
    def __init__(self):
        self.history = []
        self.filename = "translation_history.json"
        self.load_history()
    
    def add_translation(self, source_text, translated_text, source_lang, target_lang):
        """Add a translation to history"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "source_text": source_text,
            "translated_text": translated_text,
            "source_language": source_lang,
            "target_language": target_lang
        }
        self.history.append(entry)
        self.save_history()
    
    def save_history(self):
        """Save history to file"""
        try:
            with open(self.filename, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving history: {e}")
    
    def load_history(self):
        """Load history from file"""
        try:
            if os.path.exists(self.filename):
                with open(self.filename, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
        except Exception as e:
            print(f"Error loading history: {e}")
    
    def get_recent_translations(self, limit=10):
        """Get recent translations"""
        return self.history[-limit:] if self.history else []

class RealTimeTranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI-Powered Real-Time Translator")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize components
        self.translator = googletrans.Translator()
        self.lang_detector = LanguageDetector()
        self.history = TranslationHistory()
        self.speech_recognizer = sr.Recognizer()
        self.tts_engine = pyttsx3.init()
        
        # Voice settings
        self.tts_engine.setProperty('rate', 180)  # Slightly faster for clarity
        self.tts_engine.setProperty('volume', 1.0)  # Max volume
        
        # TTS queue and thread
        self.tts_queue = queue.Queue()
        self.tts_thread = threading.Thread(target=self.tts_worker, daemon=True)
        self.tts_thread.start()
        
        # State variables
        self.is_listening = False
        self.auto_detect = tk.BooleanVar(value=True)
        self.auto_speak = tk.BooleanVar(value=False)
        
        # Create GUI
        self.create_gui()
        
        # Start background processes
        self.setup_background_processing()
    
    def create_gui(self):
        """Create the main GUI"""
        
        # Title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        title_frame.pack(fill='x', pady=(0, 10))
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, text="🌍 AI Real-Time Translator", 
                              font=('Arial', 20, 'bold'), 
                              fg='white', bg='#2c3e50')
        title_label.pack(expand=True)
        
        # Main container
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Settings frame
        settings_frame = tk.LabelFrame(main_frame, text="Settings", 
                                     font=('Arial', 12, 'bold'),
                                     bg='#f0f0f0', fg='#2c3e50')
        settings_frame.pack(fill='x', pady=(0, 10))
        
        # Language selection
        lang_frame = tk.Frame(settings_frame, bg='#f0f0f0')
        lang_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(lang_frame, text="From:", font=('Arial', 10), 
                bg='#f0f0f0').pack(side='left')
        
        self.source_lang_var = tk.StringVar(value="auto")
        self.source_lang_combo = ttk.Combobox(lang_frame, textvariable=self.source_lang_var,
                                            width=15, state="readonly")
        self.source_lang_combo.pack(side='left', padx=(5, 20))
        
        tk.Label(lang_frame, text="To:", font=('Arial', 10), 
                bg='#f0f0f0').pack(side='left')
        
        self.target_lang_var = tk.StringVar(value="en")
        self.target_lang_combo = ttk.Combobox(lang_frame, textvariable=self.target_lang_var,
                                            width=15, state="readonly")
        self.target_lang_combo.pack(side='left', padx=5)
        
        # Populate language combos
        self.setup_language_combos()
        
        # Options
        options_frame = tk.Frame(settings_frame, bg='#f0f0f0')
        options_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Checkbutton(options_frame, text="Auto-detect language", 
                      variable=self.auto_detect, font=('Arial', 10),
                      bg='#f0f0f0', command=self.toggle_auto_detect).pack(side='left')
        
        tk.Checkbutton(options_frame, text="Auto-speak translation", 
                      variable=self.auto_speak, font=('Arial', 10),
                      bg='#f0f0f0').pack(side='left', padx=(20, 0))
        
        # Translation area
        translation_frame = tk.Frame(main_frame, bg='#f0f0f0')
        translation_frame.pack(fill='both', expand=True)
        
        # Input section
        input_frame = tk.LabelFrame(translation_frame, text="Input Text", 
                                  font=('Arial', 12, 'bold'),
                                  bg='#f0f0f0', fg='#2c3e50')
        input_frame.pack(fill='both', expand=True, pady=(0, 5))
        
        self.input_text = scrolledtext.ScrolledText(input_frame, height=8, 
                                                   font=('Arial', 11),
                                                   wrap=tk.WORD)
        self.input_text.pack(fill='both', expand=True, padx=10, pady=5)
        self.input_text.bind('<KeyRelease>', self.on_text_change)
        
        # Control buttons
        btn_frame = tk.Frame(input_frame, bg='#f0f0f0')
        btn_frame.pack(fill='x', padx=10, pady=5)
        
        self.voice_btn = tk.Button(btn_frame, text="🎤 Start Voice Input", 
                                  font=('Arial', 10, 'bold'),
                                  bg='#3498db', fg='white',
                                  command=self.toggle_voice_input)
        self.voice_btn.pack(side='left', padx=(0, 10))
        
        tk.Button(btn_frame, text="🔄 Translate", 
                 font=('Arial', 10, 'bold'),
                 bg='#27ae60', fg='white',
                 command=self.translate_text).pack(side='left', padx=(0, 10))
        
        tk.Button(btn_frame, text="🗑️ Clear", 
                 font=('Arial', 10),
                 bg='#e74c3c', fg='white',
                 command=self.clear_text).pack(side='left')
        
        # Output section
        output_frame = tk.LabelFrame(translation_frame, text="Translation", 
                                   font=('Arial', 12, 'bold'),
                                   bg='#f0f0f0', fg='#2c3e50')
        output_frame.pack(fill='both', expand=True, pady=(5, 0))
        
        self.output_text = scrolledtext.ScrolledText(output_frame, height=8, 
                                                    font=('Arial', 11),
                                                    wrap=tk.WORD, state='disabled')
        self.output_text.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Output control buttons
        output_btn_frame = tk.Frame(output_frame, bg='#f0f0f0')
        output_btn_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Button(output_btn_frame, text="🔊 Speak", 
                 font=('Arial', 10),
                 bg='#9b59b6', fg='white',
                 command=self.speak_translation).pack(side='left', padx=(0, 10))
        
        tk.Button(output_btn_frame, text="📋 Copy", 
                 font=('Arial', 10),
                 bg='#f39c12', fg='white',
                 command=self.copy_translation).pack(side='left', padx=(0, 10))
        
        tk.Button(output_btn_frame, text="📚 History", 
                 font=('Arial', 10),
                 bg='#34495e', fg='white',
                 command=self.show_history).pack(side='left')
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = tk.Label(self.root, textvariable=self.status_var, 
                             relief=tk.SUNKEN, anchor='w',
                             bg='#ecf0f1', font=('Arial', 9))
        status_bar.pack(side='bottom', fill='x')
    
    def setup_language_combos(self):
        """Setup language selection comboboxes"""
        # Add auto-detect option
        lang_options = ["auto (Auto-detect)"] + [f"{code} ({name.title()})" 
                                                for code, name in LANGUAGES.items()]
        
        self.source_lang_combo['values'] = lang_options
        self.target_lang_combo['values'] = lang_options[1:]  # Exclude auto for target
    
    def setup_background_processing(self):
        """Setup background processing"""
        self.translation_queue = queue.Queue()
        self.voice_queue = queue.Queue()
        
        # Start background threads
        threading.Thread(target=self.process_translations, daemon=True).start()
        # threading.Thread(target=self.process_voice_input, daemon=True).start()
    
    def toggle_auto_detect(self):
        """Toggle auto-detect functionality"""
        if self.auto_detect.get():
            self.source_lang_combo.configure(state='disabled')
        else:
            self.source_lang_combo.configure(state='readonly')
    
    def on_text_change(self, event=None):
        """Handle real-time text changes"""
        # Auto-translate after user stops typing (with delay)
        if hasattr(self, '_translate_timer'):
            self.root.after_cancel(self._translate_timer)
        
        self._translate_timer = self.root.after(1000, self.auto_translate)
    
    def auto_translate(self):
        """Auto-translate text after delay"""
        text = self.input_text.get(1.0, tk.END).strip()
        if text and len(text) > 2:  # Only auto-translate if meaningful text
            self.translate_text()
    
    def get_language_code(self, selection):
        """Extract language code from combo selection"""
        if not selection or selection.startswith("auto"):
            return "auto"
        return selection.split()[0]
    
    def translate_text(self):
        """Translate the input text"""
        text = self.input_text.get(1.0, tk.END).strip()
        if not text:
            self.update_status("No text to translate")
            return
        
        source_lang = self.get_language_code(self.source_lang_var.get())
        target_lang = self.get_language_code(self.target_lang_var.get())
        
        if source_lang == "auto" or self.auto_detect.get():
            # Use our ML model for language detection
            detected_lang = self.lang_detector.detect(text)
            source_lang = detected_lang
        
        # Queue translation
        self.translation_queue.put((text, source_lang, target_lang))
        self.update_status("Translating...")
    
    def process_translations(self):
        """Background thread to process translations"""
        while True:
            try:
                text, source_lang, target_lang = self.translation_queue.get(timeout=1)
                
                # Perform translation
                try:
                    translation = self.translator.translate(text, 
                                                          src=source_lang if source_lang != "auto" else None, 
                                                          dest=target_lang)
                    
                    # Update GUI in main thread
                    self.root.after(0, self.update_translation, 
                                  translation.text, source_lang, target_lang, text)
                    
                except Exception as e:
                    self.root.after(0, self.update_status, f"Translation error: {str(e)}")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Translation processing error: {e}")
    
    def update_translation(self, translated_text, source_lang, target_lang, original_text):
        """Update the translation display"""
        self.output_text.configure(state='normal')
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(1.0, translated_text)
        self.output_text.configure(state='disabled')
        
        # Add to history
        self.history.add_translation(original_text, translated_text, 
                                   source_lang, target_lang)
        
        # Auto-speak if enabled
        if self.auto_speak.get():
            self.speak_text(translated_text)
        
        self.update_status(f"Translated from {source_lang} to {target_lang}")
    
    def toggle_voice_input(self):
        """Toggle voice input"""
        if not self.is_listening:
            self.start_voice_input()
        else:
            self.stop_voice_input()
    
    def start_voice_input(self):
        """Start voice input"""
        self.is_listening = True
        self.voice_btn.configure(text="🔴 Stop Voice Input", bg='#e74c3c')
        self.update_status("Listening for voice input...")
        
        # Start voice recognition in background
        threading.Thread(target=self.voice_recognition_thread, daemon=True).start()
    
    def stop_voice_input(self):
        """Stop voice input"""
        self.is_listening = False
        self.voice_btn.configure(text="🎤 Start Voice Input", bg='#3498db')
        self.update_status("Voice input stopped")
    
    def voice_recognition_thread(self):
        """Background thread for voice recognition"""
        with sr.Microphone() as source:
            self.speech_recognizer.adjust_for_ambient_noise(source)
        
        while self.is_listening:
            try:
                with sr.Microphone() as source:
                    # Listen for audio with timeout
                    audio = self.speech_recognizer.listen(source, timeout=1, phrase_time_limit=5)
                
                # Recognize speech
                try:
                    text = self.speech_recognizer.recognize_google(audio)
                    self.root.after(0, self.add_voice_text, text)
                except sr.UnknownValueError:
                    pass  # Ignore unrecognized speech
                except sr.RequestError as e:
                    self.root.after(0, self.update_status, f"Voice recognition error: {e}")
                    
            except sr.WaitTimeoutError:
                pass  # Continue listening
            except Exception as e:
                print(f"Voice recognition thread error: {e}")
                break
    
    def add_voice_text(self, text):
        """Add recognized voice text to input"""
        current_text = self.input_text.get(1.0, tk.END).strip()
        if current_text:
            self.input_text.insert(tk.END, f" {text}")
        else:
            self.input_text.insert(1.0, text)
        
        self.update_status(f"Voice recognized: {text}")
        
        # Auto-translate voice input
        self.root.after(500, self.translate_text)
    
    def speak_translation(self):
        """Speak the current translation"""
        text = self.output_text.get(1.0, tk.END).strip()
        if text:
            self.speak_text(text)
    
    def speak_text(self, text):
        """Speak given text using TTS"""
        try:
            self.tts_queue.put(text)
        except Exception as e:
            self.update_status(f"TTS error: {e}")

    def tts_worker(self):
        """Background thread to process TTS queue"""
        while True:
            text = self.tts_queue.get()
            if text is None:
                break
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                self.root.after(0, self.update_status, f"TTS error: {e}")
            self.tts_queue.task_done()
    
    def copy_translation(self):
        """Copy translation to clipboard"""
        text = self.output_text.get(1.0, tk.END).strip()
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.update_status("Translation copied to clipboard")
    
    def clear_text(self):
        """Clear all text fields"""
        self.input_text.delete(1.0, tk.END)
        self.output_text.configure(state='normal')
        self.output_text.delete(1.0, tk.END)
        self.output_text.configure(state='disabled')
        self.update_status("Text cleared")
    
    def show_history(self):
        """Show translation history"""
        history_window = tk.Toplevel(self.root)
        history_window.title("Translation History")
        history_window.geometry("600x400")
        history_window.configure(bg='#f0f0f0')
        
        # History listbox
        history_frame = tk.Frame(history_window, bg='#f0f0f0')
        history_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        tk.Label(history_frame, text="Recent Translations", 
                font=('Arial', 14, 'bold'), bg='#f0f0f0').pack(pady=(0, 10))
        
        history_listbox = tk.Listbox(history_frame, font=('Arial', 10))
        history_listbox.pack(fill='both', expand=True)
        
        # Populate history
        recent_translations = self.history.get_recent_translations(20)
        for entry in reversed(recent_translations):
            timestamp = datetime.fromisoformat(entry['timestamp']).strftime("%m/%d %H:%M")
            display_text = f"[{timestamp}] {entry['source_text'][:30]}... → {entry['translated_text'][:30]}..."
            history_listbox.insert(0, display_text)
    
    def update_status(self, message):
        """Update status bar"""
        self.status_var.set(message)
        self.root.update_idletasks()

def main():
    """Main function to run the application"""
    # Check for required dependencies
    try:
        import speech_recognition
        import pyttsx3
        import googletrans
        import pyaudio
        import sklearn
    except ImportError as e:
        print(f"Missing required dependency: {e}")
        print("Please install required packages:")
        print("pip install SpeechRecognition pyttsx3 googletrans==4.0.0rc1 pyaudio scikit-learn")
        return
    
    # Create and run the application
    root = tk.Tk()
    app = RealTimeTranslatorApp(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nApplication closed by user")
    except Exception as e:
        print(f"Application error: {e}")

if __name__ == "__main__":
    main()