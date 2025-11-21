import streamlit as st
import torch
import os
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib

# --------------------------
# Correct Paths (based on your screenshot)
# --------------------------
MODEL_PATH = "notebooks/models/distilbert_lora"
ENCODER_PATH = "notebooks/models/label_encoder.pkl"

# We'll load model/tokenizer/encoder lazily to avoid blocking Streamlit
tokenizer = None
model = None
le = None

def load_resources():
    """Attempt to load tokenizer, model and label encoder. Return tuple(success, message)."""
    global tokenizer, model, le
    try:
        # Load label encoder first to determine number of classes
        if le is None:
            le = joblib.load(ENCODER_PATH)

        num_labels = len(le.classes_)

        # Load tokenizer if needed
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

        # If this model folder contains a PEFT/LoRA adapter, load base model with
        # correct num_labels then apply the adapter using `peft`.
        adapter_config_path = os.path.join(MODEL_PATH, "adapter_config.json")
        adapter_weights_path = os.path.join(MODEL_PATH, "adapter_model.safetensors")
        has_adapter = os.path.exists(adapter_config_path) or os.path.exists(adapter_weights_path)

        if has_adapter:
            # Try to load base model name from adapter config if present
            base_model_name = None
            try:
                if os.path.exists(adapter_config_path):
                    with open(adapter_config_path, "r", encoding="utf-8") as f:
                        ac = json.load(f)
                        base_model_name = ac.get("base_model_name_or_path")
            except Exception:
                base_model_name = None

            base_to_load = base_model_name or "distilbert-base-uncased"

            # Load base model with the same number of labels as the adapter expects
            base_model = AutoModelForSequenceClassification.from_pretrained(base_to_load, num_labels=num_labels)

            try:
                from peft import PeftModel
            except Exception as e:
                return False, (
                    "PEFT package is required to load LoRA adapters. Install with: `pip install peft`\n"
                    f"Import error: {e}"
                )

            model = PeftModel.from_pretrained(base_model, MODEL_PATH, is_train=False)
            model.eval()
        else:
            # No adapter: load model directly from MODEL_PATH with correct labels
            if model is None:
                model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=num_labels)
                model.eval()

        return True, ""
    except Exception as e:
        # Return the exception message so the UI can show it.
        return False, str(e)


def predict_driver(text):
    ok, msg = load_resources()
    if not ok:
        raise RuntimeError(f"Failed loading model resources: {msg}")

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256
    )
    with torch.no_grad():
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=1).item()
    return le.inverse_transform([pred])[0]

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="F1 Winner Predictor", page_icon="🏎️", layout="centered")

st.title("🏎️ F1 Winner Prediction App")
st.write("Enter a race summary and get predicted winner (DistilBERT + LoRA).")

text_input = st.text_area("Enter Race Summary:", height=250)

if st.button("Predict Winner"):
    if text_input.strip() == "":
        st.warning("Please enter text.")
    else:
        with st.spinner("Predicting..."):
            try:
                result = predict_driver(text_input)
            except Exception as e:
                # Surface load/prediction errors in the UI instead of failing silently
                st.error(f"Error: {e}")
            else:
                st.success(f"🏆 Predicted Winner: **{result}**")
