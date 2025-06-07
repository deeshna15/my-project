
'''from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
import re

app = Flask(__name__)

# Load model and tokenizer
model = load_model("fake_news_lstm_model.h5")
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

max_len = 200

# Text cleaning function (same logic as training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# Prediction function
def predict_news(text):
    cleaned_text = clean_text(text)
    print("Cleaned Text:", cleaned_text)
    
    seq = tokenizer.texts_to_sequences([cleaned_text])
    print("Tokenized Sequence:", seq)

    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    print("Padded Sequence:", padded)

    prediction = model.predict(padded)
    print("Model Raw Prediction:", prediction)

    return "True" if prediction[0][0] >= 0.5 else "Fake", round(float(prediction[0][0]), 2)

# Route
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    if request.method == "POST":
        text = request.form["news"]
        result, confidence = predict_news(text)
    return render_template("index.html", result=result, confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)'''
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
import re
import numpy as np

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load model and tokenizer
model = load_model("fake_news_model.h5")
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

max_len = 54

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def predict_news(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    prediction = model.predict(padded)
    return ("True", round(float(prediction[0][0]), 2)) if prediction[0][0] >= 0.5 else ("Fake", round(float(prediction[0][0]), 2))

@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None, "confidence": None})

@app.post("/", response_class=HTMLResponse)
async def form_post(request: Request, news: str = Form(...)):
    result, confidence = predict_news(news)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result,
        "confidence": confidence
    })
