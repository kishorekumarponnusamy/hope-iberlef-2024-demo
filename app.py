import streamlit as st
from transformers import pipeline

MODEL_ID = "kishorekumarponnusamy/HOPE_IberLEF_2024"
SUBFOLDER = "bert-base-spanish-wwm-cased"

@st.cache_resource
def load_model():
    clf = pipeline(
        "text-classification",
        model=MODEL_ID,
        tokenizer=MODEL_ID,
        model_kwargs={"subfolder": SUBFOLDER},
        tokenizer_kwargs={"subfolder": SUBFOLDER}
    )
    return clf

classifier = load_model()

st.title("HOPE IberLEF 2024 Demo")

text = st.text_area("Enter Spanish text")

label_map = {
    "LABEL_0": "Non-Hope Speech",
    "LABEL_1": "Hope Speech"
}

if st.button("Predict"):
    if text.strip():
        result = classifier(text)[0]
        label = result["label"]
        score = result["score"]

        st.write("Prediction:", label_map.get(label, label))
        st.write("Confidence:", round(score, 4))
    else:
        st.warning("Enter text")
