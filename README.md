🏎️ **F1 Race Winner Prediction using NLP + DistilBERT + LoRA**

# 🏎️ Formula 1 Race Winner Prediction using NLP & DistilBERT (LoRA Finetuning)

This project predicts **the winner of a Formula 1 Grand Prix** using **text-based race summaries**, practice reports, and weekend analysis.
It applies modern **NLP techniques**, **data balancing**, and **LoRA-based transformer finetuning** to classify which driver is most likely to win.

---

# 📌 **Project Motivation**

F1 race summaries contain rich contextual information:

* Driver performance
* Team upgrades
* Practice/qualifying pace
* Weather
* Tyre degradation
* Car balance/handling
* Engineering analysis

A neural network can learn these patterns and predict the likely race winner.

---

# 🚀 **Tech Stack**

| Component  | Technology                                 |
| ---------- | ------------------------------------------ |
| Language   | Python                                     |
| NLP Model  | DistilBERT (LoRA Finetuning)               |
| Finetuning | PEFT (LoRA)                                |
| Dataset    | Custom F1 text dataset (2020–2025)         |
| UI         | Streamlit Web App                          |
| Training   | CPU-friendly (no GPU required)             |
| Libraries  | Transformers, PEFT, Datasets, Scikit-learn |

---

# 🗂️ **Folder Structure**

```
NLP PROJECT/
│
├── Data/
│   ├── f1_dataset.csv
│   ├── balanced_f1_dataset.csv
│   └── (raw data files)
│
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 01.0_balancing_dataset.ipynb
│   ├── 02_modeltraining.ipynb
│   ├── 03_prediction.ipynb
│   ├── 04_finetuning.ipynb
│   ├── 05_app.ipynb
│   └── models/
│       ├── distilbert_lora/
│       │     ├── config.json
│       │     ├── pytorch_model.bin
│       │     ├── adapter_config.json
│       │     └── other tokenizer/model files
│       ├── bert_model.pt
│       └── label_encoder.pkl
│
├── app.py                      # Streamlit App
├── README.md                   # You are here
└── requirements.txt
```

---

# 📚 **Dataset Preparation**

### 1️⃣ **Raw Data**

Collected F1 race weekend summaries (FP1/FP2/FP3, Qualifying, Race reports) from 2020–2024.

### 2️⃣ **Cleaning**

* Lowercasing
* Removing special characters
* Removing extra whitespace
* Standardizing driver/team names

### 3️⃣ **Label Encoding**

Winners were encoded using:

```python
LabelEncoder()
```

### 4️⃣ **Major Issue: Class Imbalance**

Most races were won by:

* Max Verstappen
* Lewis Hamilton

Other drivers (Norris, Russell, Leclerc, Piastri) were heavily under-represented.

### 5️⃣ **Balancing**

A custom balancing script:

* Oversampled rare winners
* Undersampled majority winners
* Resulted in equal samples per driver
* Fixed strong prediction bias

Final dataset stored as:

```
balanced_f1_dataset.csv
```

---

# 🧠 **Model Architecture**

## 🎯 1. Baseline: TF-IDF + Dense NN

(Built initially for testing)

## 🎯 2. Transformer Models Tried

### ❌ Qwen2-0.5B + QLoRA

* Too slow on CPU
* BitsAndBytes 4-bit layers failed backward pass
* Not suitable for Windows-CPU environment

### ✅ Final Model: **DistilBERT + LoRA Finetuning**

Chosen because:

* Lightweight
* Very fast on CPU
* Finetunes quickly (~10–20 seconds)
* Excellent performance
* Works perfectly with LoRA adapters (PEFT)

---

# 🔧 **Training Pipeline (DistilBERT + LoRA)**

### ✔ Tokenization (256 max length)

### ✔ LoRA adapters added to DistilBERT attention layers

### ✔ Optimizer: AdamW

### ✔ Batch size: 8

### ✔ Epochs: 4

### ✔ Loss function: Cross Entropy

### ✔ PEFT saves only LoRA weights (very small)

Training notebook:

```
04_finetuning.ipynb
```

---

# 🧪 **Prediction Pipeline**

Prediction steps:

1. Load tokenizer + LoRA finetuned DistilBERT
2. Encode input text
3. Pass through classifier
4. Convert label → driver using stored label encoder

Notebook for predictions:

```
03_prediction.ipynb
```

---

# 🌐 **Streamlit App**

File:

```
app.py
```

Features:

* Text box for entering race weekend summary
* Predict button
* Outputs winning driver
* Shows clear UI messages

Run using:

```
streamlit run app.py
```

---

# 📈 **Results**

### After dataset balancing + LoRA finetuning:

* Predictions became **more fair**
* Less biased toward Verstappen/Hamilton
* Model correctly identifies strong patterns for:

  * Lando Norris
  * Charles Leclerc
  * George Russell
  * Oscar Piastri
  * Sergio Pérez

### Accuracy improved significantly

(based on balanced dataset split)

---

# 🧭 **Key Learnings**

* Raw transformer finetuning can be slow on CPU
* QLoRA needs GPU to function properly
* DistilBERT is very powerful for small datasets
* Dataset balancing is **critical** in motorsport NLP
* LoRA drastically reduces training time
* Streamlit brings the model to life with a clean UI

---

# 🚀 **How to Run the Project**

### 1. Clone Repo

```
git clone https://github.com/<your-username>/f1-nlp-winner-prediction.git
cd f1-nlp-winner-prediction
```

### 2. Install Requirements

```
pip install -r requirements.txt
```

### 3. Run Preprocessing (optional)

```
jupyter notebook notebooks/01_preprocessing.ipynb
```

### 4. Train Model

```
jupyter notebook notebooks/04_finetuning.ipynb
```

### 5. Run Streamlit App

```
streamlit run app.py
```



# ✨ **Author**

**Shrivatsh and Pranav**
Mahindra University
NLP Project — Formula 1 Winner Prediction

---


