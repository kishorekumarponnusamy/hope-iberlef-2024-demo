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

# --- Show model label mapping (important debugging) ---
st.subheader("Model Label Info")
st.write(classifier.model.config.id2label)

text = st.text_area("Enter Spanish text")

# Try BOTH mappings (you will verify which is correct)
label_map = {
    "LABEL_0": "Non-Hope Speech",
    "LABEL_1": "Hope Speech"
}

# Alternative (uncomment if reversed)
# label_map = {
#     "LABEL_0": "Hope Speech",
#     "LABEL_1": "Non-Hope Speech"
# }

if st.button("Predict"):
    if text.strip():
        
        # Get probabilities for all classes
        results = classifier(text, return_all_scores=True)[0]

        st.subheader("Class Probabilities")

        for r in results:
            st.write(
                label_map.get(r["label"], r["label"]),
                ":", round(r["score"], 4)
            )

        # Select best prediction
        best = max(results, key=lambda x: x["score"])

        st.subheader("Final Prediction")
        st.write("Prediction:", label_map.get(best["label"], best["label"]))
        st.write("Confidence:", round(best["score"], 4))
    else:
        st.warning("Enter text")
