# HOPE IberLEF 2024 Demo 🇪🇸
<p align="center">
  <img src="./image/workflow.png" alt="Workflow" width="800">
</p>
A deep learning model for Spanish Hope Speech Detection using a fine-tuned BERT-based transformer model, deployed through an interactive Streamlit web application.

---

## 📌 About the Project

This project detects Hope Speech and Non-Hope Speech in Spanish text. The model was developed for the HOPE IberLEF 2024 shared task and uses a fine-tuned `bert-base-spanish-wwm-cased` transformer model.

The web application allows users to enter Spanish text and obtain a real-time classification result with confidence scores.

---

## ✨ Features

- 🔍 Detects Hope Speech in Spanish text
- 🤖 Powered by a fine-tuned BERT transformer model
- 🌐 Interactive Streamlit web application
- 📊 Shows class probabilities and prediction confidence
- 📁 Loads the model directly from Hugging Face

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-orange)
![BERT](https://img.shields.io/badge/Model-BERT-green)

---

## 📂 Project Structure

```text
hope-iberlef-2024-demo/
├── app.py                 # Streamlit web application
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## 🚀 How to Run
1. Clone the repository
```
git clone https://github.com/kishorekumarponnusamy/hope-iberlef-2024-demo.git
cd hope-iberlef-2024-demo
```
2. Install dependencies
```
pip install -r requirements.txt
```
3. Run the Streamlit app
```
streamlit run app.py
```
## 🤗 Hugging Face Model

The model is hosted on Hugging Face:
```
kishorekumarponnusamy/HOPE_IberLEF_2024
```
Model subfolder:
```
bert-base-spanish-wwm-cased
```
## 🌐 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hope-iberlef-2024-demo-o64uq9takqyyzwa4rxhz3e.streamlit.app/)

> Tip: Click the badge to open the app in a new tab.

## 🧪 Example Input
```
La población de origen inmigrante ocupa en mayor medida puestos de trabajo en los sectores más precarizados.
```
## 📌 Output

The application returns:

- Predicted class
- Confidence score
- Class probability distribution
