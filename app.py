import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models

# ==========================================
# 1. PAGE CONFIGURATION & MEDICAL STYLING
# ==========================================
st.set_page_config(
    page_title="NeuroScan AI | Brain MRI Tumor Diagnostic Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Medical Dashboard CSS
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-box {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 12px;
        padding: 20px;
        color: #ffffff;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        margin-bottom: 15px;
    }
    .badge-danger {
        background-color: #ef4444;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 14px;
        display: inline-block;
    }
    .badge-success {
        background-color: #10b981;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 14px;
        display: inline-block;
    }
    .badge-warning {
        background-color: #f59e0b;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 14px;
        display: inline-block;
    }
    .stProgress > div > div > div > div {
        background-color: #3b82f6;
    }
    .clinical-disclaimer {
        font-size: 12px;
        color: #94a3b8;
        border-top: 1px solid #334155;
        padding-top: 10px;
        margin-top: 25px;
    }
</style>
""", unsafe_allow_html=True)

# Diagnostic Class Mappings
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
CLASS_LABELS = {
    'glioma': 'Glioma Tumor',
    'meningioma': 'Meningioma Tumor',
    'pituitary': 'Pituitary Adenoma',
    'notumor': 'Healthy (No Tumor Detected)'
}
CLASS_DESCRIPTIONS = {
    'glioma': "A primary intracranial tumor originating in neuroglial stem or progenitor cells. Often requires urgent neurosurgical oncology evaluation.",
    'meningioma': "Typically slow-growing, benign extra-axial neoplasm arising from the arachnoid cap cells of the meninges.",
    'pituitary': "Endocrine neoplasm arising in the anterior pituitary gland within the sella turcica, potentially impacting hormone axes and chiasm.",
    'notumor': "Brain parenchyma presents normal physiological morphology and baseline tissue contrast without evident mass lesions."
}

# ==========================================
# 2. MODEL ARCHITECTURE & GRAD-CAM
# ==========================================
class MedicalBrainResNet50(nn.Module):
    def __init__(self, num_classes=4):
        super(MedicalBrainResNet50, self).__init__()
        self.backbone = models.resnet50(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, x, class_idx=None):
        self.model.eval()
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(self.activations, dim=1).squeeze().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        return heatmap, class_idx, torch.softmax(output, dim=1).detach().cpu().numpy()[0]

# ==========================================
# 3. PREPROCESSING UTILITIES
# ==========================================
def crop_brain_contour(image_np):
    """Extreme-point contour cropping to isolate cerebral parenchyma."""
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return image_np
    
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 1000:
        return image_np
    
    ext_left = tuple(c[c[:, :, 0].argmin()][0])
    ext_right = tuple(c[c[:, :, 0].argmax()][0])
    ext_top = tuple(c[c[:, :, 1].argmin()][0])
    ext_bot = tuple(c[c[:, :, 1].argmax()][0])
    
    cropped = image_np[ext_top[1]:ext_bot[1], ext_left[0]:ext_right[0]]
    return cropped if cropped.size > 0 else image_np

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

@st.cache_resource
def load_trained_model(weights_path="best_brain_tumor_resnet50.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MedicalBrainResNet50(num_classes=4)
    
    weights_loaded = False
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        weights_loaded = True
    else:
        # Fallback to pretrained resnet backbone if fine-tuned weights not found yet
        pretrained_bb = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Copy backbone layers
        bb_dict = {k: v for k, v in pretrained_bb.state_dict().items() if 'fc' not in k}
        model_state = model.backbone.state_dict()
        model_state.update(bb_dict)
        model.backbone.load_state_dict(model_state, strict=False)

    model = model.to(device)
    model.eval()
    grad_cam = GradCAM(model, model.backbone.layer4)
    return model, grad_cam, device, weights_loaded

# ==========================================
# 4. SIDEBAR CONFIGURATION
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2864/2864380.png", width=70)
    st.title("NeuroScan AI")
    st.caption("Deep Learning Brain MRI Classifier")
    
    model, grad_cam, device, weights_loaded = load_trained_model()
    
    st.markdown("---")
    st.subheader("System Status")
    if weights_loaded:
        st.success("Custom ResNet-50 Weights Loaded (`best_brain_tumor_resnet50.pth`)")
    else:
        st.warning("Running with ImageNet Pretrained Backbone (Train notebook to export custom `.pth`)")
    
    st.info(f"Compute Engine: `{str(device).upper()}`")
    
    st.markdown("---")
    st.subheader("Clinical Filters")
    enable_crop = st.toggle("Automated Contour Brain Cropping", value=True, help="Eliminates black scanner borders and non-cerebral artifacts.")
    show_gradcam = st.toggle("Explainable AI (Grad-CAM Saliency)", value=True, help="Overlays heatmaps highlighting regions influencing the model's decision.")
    cam_opacity = st.slider("Grad-CAM Overlay Intensity", min_value=0.1, max_value=0.9, value=0.45, step=0.05)

# ==========================================
# 5. MAIN INTERFACE
# ==========================================
st.title("🧠 Medical Brain MRI Oncology Diagnostic Assistant")
st.write(
    "Upload a patient's **axial, coronal, or sagittal T1/T2/FLAIR Brain MRI slice** to screen for intracranial neoplasms "
    "(Glioma, Meningioma, Pituitary Adenoma) versus Healthy parenchyma."
)

uploaded_file = st.file_uploader(
    "Choose a Brain MRI scan (JPEG / PNG)",
    type=["jpg", "jpeg", "png"],
    help="Upload a standard high-contrast 2D brain MRI scan."
)

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Optional Preprocessing
    processed_rgb = crop_brain_contour(image_rgb) if enable_crop else image_rgb
    pil_img = Image.fromarray(processed_rgb)

    # Tensor Transformation & Inference
    transform = get_transforms()
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    # Run Prediction & Grad-CAM
    heatmap, pred_idx, probabilities = grad_cam.generate(input_tensor)
    predicted_slug = CLASSES[pred_idx]
    predicted_name = CLASS_LABELS[predicted_slug]
    confidence_val = probabilities[pred_idx] * 100
    is_cancer = predicted_slug != 'notumor'

    # Saliency Map Generation
    resized_heatmap = cv2.resize(heatmap, (processed_rgb.shape[1], processed_rgb.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * resized_heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    gradcam_overlay = np.float32(heatmap_colored) / 255 * cam_opacity + np.float32(processed_rgb) / 255 * (1 - cam_opacity)
    gradcam_overlay = np.clip(gradcam_overlay, 0, 1)

    st.markdown("---")

    # Diagnostic Summary Alert
    col_metric, col_badge = st.columns([3, 1])
    with col_metric:
        if is_cancer:
            st.error(f"### ⚠️ Brain Tumor Detected: **{predicted_name}**")
        else:
            st.success(f"### ✅ Negative for Intracranial Mass: **{predicted_name}**")
        st.write(CLASS_DESCRIPTIONS[predicted_slug])

    with col_badge:
        st.write("**Diagnostic Confidence**")
        if confidence_val >= 85:
            st.markdown(f"<span class='badge-danger' style='background-color:#10b981;'>High: {confidence_val:.1f}%</span>", unsafe_allow_html=True)
        elif confidence_val >= 60:
            st.markdown(f"<span class='badge-warning'>Moderate: {confidence_val:.1f}%</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span class='badge-danger'>Inconclusive: {confidence_val:.1f}%</span>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Image Visualization Columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("1. Raw MRI Scan")
        st.image(image_rgb, use_container_width=True, caption=f"Original Size: {image_rgb.shape[1]}x{image_rgb.shape[0]}px")

    with col2:
        st.subheader("2. Preprocessed Parenchyma")
        st.image(processed_rgb, use_container_width=True, caption="Extreme-Point Skull Stripped / Cropped")

    with col3:
        st.subheader("3. Grad-CAM Lesion Focus")
        if show_gradcam:
            st.image(gradcam_overlay, use_container_width=True, caption="Neural Attention Heatmap Overlay")
        else:
            st.info("Grad-CAM visualization toggled off in sidebar.")

    # Differential Diagnosis Probabilities Table
    st.markdown("### 📊 Differential Diagnosis Breakdown")
    prob_cols = st.columns(4)
    
    for i, cls_name in enumerate(CLASSES):
        prob = probabilities[i] * 100
        with prob_cols[i]:
            card_class = "badge-danger" if cls_name == predicted_slug and is_cancer else ("badge-success" if cls_name == predicted_slug else "badge-warning")
            st.markdown(f"""
            <div class="metric-box">
                <div style="font-size:14px; color:#cbd5e1; margin-bottom:6px;">{CLASS_LABELS[cls_name]}</div>
                <div style="font-size:24px; font-weight:bold; margin-bottom:10px;">{prob:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(float(probabilities[i]))

else:
    # Initial landing instructions when no file is uploaded yet
    st.info("👆 Please upload a Brain MRI image above to start automated cancer detection.")
    
    st.markdown("### 🔍 Supported Diagnostic Categories")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**Glioma**\nAstrocytomas, Oligodendrogliomas, and Glioblastomas.")
    with c2:
        st.markdown("**Meningioma**\nDural-based masses arising from arachnoid membranes.")
    with c3:
        st.markdown("**Pituitary Adenoma**\nSellar and suprasellar tumors affecting pituitary tissues.")
    with c4:
        st.markdown("**No Tumor (Normal)**\nUnremarkable cerebral cortex and ventricles.")

st.markdown("""
<div class="clinical-disclaimer">
    <strong>⚠️ Clinical Disclaimer:</strong> This artificial intelligence system is intended strictly as an investigational research decision-support tool. 
    It is not certified as an independent medical device. All outputs and segmentation masks must be reviewed and correlated with histological, clinical, and radiological findings by a certified neuro-radiologist or medical doctor.
</div>
""", unsafe_allow_html=True)
