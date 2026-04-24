import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import SwinModel
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Lung Cancer Classifier", page_icon="🫁", layout="centered")

# ── Constants ────────────────────────────────────────────────────────────────
# Must match the sorted canonical_classes from the notebook exactly
CANONICAL_CLASSES = [
    "adenocarcinoma",
    "large.cell.carcinoma",
    "malignant_cases",
    "normal",
    "squamous.cell.carcinoma",
]

DISPLAY_NAMES = {
    "adenocarcinoma": "Adenocarcinoma",
    "large.cell.carcinoma": "Large Cell Carcinoma",
    "malignant_cases": "Malignant (Other)",
    "normal": "Normal / Benign",
    "squamous.cell.carcinoma": "Squamous Cell Carcinoma",
}

MODEL_REPO = "Pranav379/lung_cancer_model"
MODEL_FILENAME = "model2a_best.pth"

# Model definition
class SwinClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = SwinModel.from_pretrained(
            "microsoft/swin-tiny-patch4-window7-224"
        )
        hidden = self.backbone.config.hidden_size
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x):
        out = self.backbone(pixel_values=x)
        x = out.last_hidden_state[:, 0]
        return self.head(x)


# ── Preprocessing (test_transform from notebook, no augmentation) ────────────
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),   # CT scans → 3-ch grayscale
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# Load model
@st.cache_resource
def load_model():
    import os
    from huggingface_hub import hf_hub_download

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Download model
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,
        token=os.getenv("HF_TOKEN")
    )

    model = SwinClassifier(num_classes=len(CANONICAL_CLASSES))
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return model, device


# ── Inference helper ─────────────────────────────────────────────────────────
def predict(image: Image.Image, model, device):
    tensor = test_transform(image).unsqueeze(0).to(device)   # (1, 3, 224, 224)
    with torch.no_grad():
        logits = model(tensor)                                # (1, num_classes)
        probs = torch.softmax(logits, dim=1).squeeze()       # (num_classes,)
    return probs.cpu().numpy()


# Dashboard
st.title("🫁 Lung Cancer CT Classifier")
st.markdown(
    "Upload a lung CT scan image and the model will classify it into one of "
    "five categories using a fine-tuned **Swin Transformer**."
)

uploaded_file = st.file_uploader(
    "Choose a CT scan image", type=["jpg", "jpeg", "png", "bmp", "tiff"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with col2:
        with st.spinner("Loading model and running inference…"):
            try:
                model, device = load_model()
                probs = predict(image, model, device)
            except FileNotFoundError:
                st.error(
                    f"⚠️ Model file `{MODEL_PATH}` not found. "
                    "Place it in the same directory as `app.py` and restart."
                )
                st.stop()

        top_idx = int(np.argmax(probs))
        top_class = CANONICAL_CLASSES[top_idx]
        top_prob = probs[top_idx]

        st.subheader("Prediction")
        color = "green" if top_class == "normal" else "red"
        st.markdown(
            f"<h3 style='color:{color}'>{DISPLAY_NAMES[top_class]}</h3>",
            unsafe_allow_html=True,
        )
        st.metric("Confidence", f"{top_prob * 100:.1f}%")

        st.subheader("All class probabilities")
        for cls, prob in sorted(
            zip(CANONICAL_CLASSES, probs), key=lambda x: -x[1]
        ):
            st.progress(
                float(prob),
                text=f"{DISPLAY_NAMES[cls]}: {prob * 100:.1f}%",
            )

st.divider()
st.caption(
    "Dataset: Lung Cancer CT Scan Images - https://www.kaggle.com/datasets/dishantrathi20/ct-scan-images-for-lung-cancer"
)
