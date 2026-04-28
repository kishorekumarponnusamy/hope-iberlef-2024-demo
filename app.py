%%writefile app.py

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "kishorekumarponnusamy/HOPE_IberLEF_2024"
SUBFOLDER = "bert-base-spanish-wwm-cased"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        subfolder=SUBFOLDER,
        use_fast=False
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        subfolder=SUBFOLDER
    )

    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

st.title("HOPE IberLEF 2024 Demo")
st.write("Spanish Hope Speech Classification")

text = st.text_area("Enter Spanish text")

label_map = {
    0: "Non-Hope Speech",
    1: "Hope Speech"
}

if st.button("Predict"):
    if text.strip():
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        st.subheader("Result")
        st.write("Prediction:", label_map.get(pred, f"Class {pred}"))
        st.write("Confidence:", round(float(probs[0][pred]), 4))
    else:
        st.warning("Enter text")