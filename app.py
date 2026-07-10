import os
# --- Must set before importing tensorflow ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # disable GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # reduce TF logging

import logging
from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# NLTK: download only if necessary
try:
    _ = stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

try:
    # punkt for word_tokenize
    _ = word_tokenize("test")
except LookupError:
    nltk.download('punkt')

# App setup
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths (adjust as needed)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'saved_models', 'fake_news_detector.h5')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'models', 'saved_models', 'tokenizer.pickle')

# Model / Tokenizer load with error handling
model = None
tokenizer = None
try:
    model = load_model(MODEL_PATH)
    logger.info("Model loaded from %s", MODEL_PATH)
except Exception as e:
    logger.exception("Failed to load model. Check MODEL_PATH: %s", MODEL_PATH)
    raise

try:
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    logger.info("Tokenizer loaded from %s", TOKENIZER_PATH)
except Exception as e:
    logger.exception("Failed to load tokenizer. Check TOKENIZER_PATH: %s", TOKENIZER_PATH)
    raise

# Preprocessing function
STOP_WORDS = set(stopwords.words('english'))

def preprocess_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    # remove urls and html
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'<.*?>', '', text)
    # keep only letters and spaces
    text = re.sub(r"[^a-zA-Z\s]", '', text)
    text = text.strip()
    # tokenize and remove stopwords
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS]
    return ' '.join(tokens)

# Prediction settings
MAX_LEN = 100   # MUST match length used when training
THRESHOLD = 0.5

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form.get('text', '')
    if not input_text.strip():
        return render_template('index.html', prediction="Please enter text to analyze.")

    cleaned_text = preprocess_text(input_text)
    if not cleaned_text:
        return render_template('index.html', prediction="Text empty after preprocessing.")

    sequences = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

    # model.predict returns an array; get scalar
    pred_array = model.predict(padded)
    # Support different model output shapes
    try:
        pred = float(pred_array.ravel()[0])
    except Exception:
        pred = float(pred_array[0])

    result = "Real News" if pred >= THRESHOLD else "Fake News"
    confidence_pct = round(pred * 100, 2)

    return render_template('index.html', prediction=f"{result} (confidence: {confidence_pct}%)")

if __name__ == '__main__':
    # Warning: do NOT use debug=True in production
    app.run(debug=True, host='0.0.0.0', port=5000)
