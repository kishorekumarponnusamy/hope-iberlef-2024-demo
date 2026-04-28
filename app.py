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
        tokenizer_kwargs={"subfolder": SUBFOLDER, "use_fast": False}
    )
    return clf

classifier = load_model()

st.title("HOPE IberLEF 2024 Demo")

st.subheader("Model Label Info")
st.write(classifier.model.config.id2label)

text = st.text_area("Enter Spanish text")

label_map = {
    "LABEL_0": "Non-Hope Speech",
    "LABEL_1": "Hope Speech"
}

if st.button("Predict"):
    if text.strip():

        results = classifier(text, top_k=None)

        if isinstance(results[0], list):
            results = results[0]

        st.subheader("Class Probabilities")

        for r in results:
            label = r["label"]
            score = r["score"]
            st.write(f"{label_map.get(label, label)}: {score:.4f}")

        best = max(results, key=lambda x: x["score"])

        st.subheader("Final Prediction")
        st.write("Prediction:", label_map.get(best["label"], best["label"]))
        st.write("Confidence:", round(best["score"], 4))

    else:
        st.warning("Enter text")
