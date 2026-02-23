import streamlit as st
import sqlite3
import hashlib
import numpy as np
import datetime
import tensorflow as tf
import gdown
import os

from PIL import Image
from tensorflow.keras.preprocessing import image

# ==============================
# MODEL + CONSTANTS
# ==============================

IMG_SIZE = 224

CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

CLASS_INFO = {
    'akiec': {'name': 'Actinic Keratosis', 'type': 'Precancer'},
    'bcc': {'name': 'Basal Cell Carcinoma', 'type': 'Cancer'},
    'bkl': {'name': 'Benign Keratosis', 'type': 'Benign'},
    'df': {'name': 'Dermatofibroma', 'type': 'Benign'},
    'mel': {'name': 'Melanoma', 'type': 'Cancer'},
    'nv': {'name': 'Melanocytic Nevus', 'type': 'Benign'},
    'vasc': {'name': 'Vascular Lesion', 'type': 'Benign'}
}

MODEL_PATH = "skin_cancer_model.h5"
DRIVE_FILE_ID = "1orMb-xYmIEfxoLbqRKw11TFk7euPs0gk"

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(page_title="🩺 Skin Cancer AI Classifier", layout="wide")  

# ==============================
# DATABASE
# ==============================
conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS users(
    username TEXT PRIMARY KEY,
    password TEXT
)
""")
conn.commit()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                  (username, hash_password(password)))
        conn.commit()
        return True
    except:
        return False

def login_user(username, password):
    c.execute("SELECT * FROM users WHERE username=? AND password=?",
              (username, hash_password(password)))
    return c.fetchone()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None

# ==============================
# 🎨 AUTH PAGE
# ==============================
# ==============================
# 🎨 FIXED CLEAN LOGIN PAGE
# ==============================
def auth_page():
    st.markdown("""
    <style>
    /* Hide header & footer */
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Page background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }

    /* Remove default block padding */
    .block-container {
        padding-top: 4rem !important;
        padding-bottom: 0rem !important;
    }

    /* Card style applied to the center column */
    [data-testid="column"]:nth-child(2) {
        background: #ffffff;
        padding: 40px 35px !important;
        border-radius: 20px;
        box-shadow: 0 15px 40px rgba(0,0,0,0.12);
    }

    /* Inputs */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 1.5px solid #e0e0e0;
        padding: 10px 14px;
        font-size: 15px;
    }
    .stTextInput > div > div > input:focus {
        border-color: #4A90E2;
        box-shadow: 0 0 0 2px rgba(74,144,226,0.2);
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(90deg, #4A90E2, #6C63FF);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px;
        font-size: 16px;
        font-weight: 600;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover {
        opacity: 0.9;
    }

    /* Radio buttons */
    .stRadio > div {
        justify-content: center;
        gap: 20px;
    }
    .stRadio label {
        font-size: 15px;
        font-weight: 500;
    }

    /* Hide the ugly radio container border */
    .stRadio > div[role="radiogroup"] {
        flex-direction: row;
        gap: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Use 3 columns — center one acts as the card
    left, center, right = st.columns([1, 1.2, 1])

    with center:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<h2 style='text-align:center; color:#2d2d2d; margin-bottom:8px;'>🔐 Secure Login</h2>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<p style='text-align:center; color:#888; font-size:14px; margin-bottom:20px;'>Welcome back! Please login to continue.</p>",
            unsafe_allow_html=True
        )

        choice = st.radio("", ["Login", "Register"], horizontal=True, label_visibility="collapsed")

        st.markdown("<br>", unsafe_allow_html=True)

        if choice == "Login":
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Login", use_container_width=True):
                result = login_user(username, password)
                if result:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("❌ Invalid Username or Password")

        else:
            new_user = st.text_input("Choose Username", placeholder="Pick a username")
            new_pass = st.text_input("Choose Password", type="password", placeholder="Pick a password")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Register", use_container_width=True):
                if register_user(new_user, new_pass):
                    st.success("✅ Account Created Successfully! Please login.")
                else:
                    st.error("❌ Username already exists")


# STOP IF NOT LOGGED IN
if not st.session_state.logged_in:
    auth_page()
    st.stop()

# ==============================
# SIDEBAR
# ==============================
st.sidebar.markdown(f"👋 Logged in as: *{st.session_state.username}*")
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.username = None
    st.rerun()

st.sidebar.title("📊 Model Overview")
st.sidebar.metric("Model", "MobileNetV2")
st.sidebar.metric("Accuracy", "91%")
st.sidebar.metric("Validation Accuracy", "88%")
st.sidebar.metric("Classes", "7")

# ==============================
# MAIN HEADER (After Login)
# ==============================
st.markdown("""
<div style="
background: linear-gradient(90deg,#0f2027,#203a43,#2c5364);
padding: 35px;
border-radius: 20px;
text-align: center;
color: white;
margin-bottom: 40px;">
<h1>🩺 Skin Cancer AI Classifier</h1>
<h3>Early Detection | Rural Healthcare Support | Instant AI Prediction</h3>
</div>
""", unsafe_allow_html=True)

# बाकी तुझा prediction code जसाच्या तसा खाली सुरू राहील...

# ===== WHY THIS MATTERS =====
st.markdown("## 🇮🇳 Why This Matters")
st.info("""
• Rising skin cancer cases in India  
• Rural areas lack dermatology specialists  
• Early detection improves survival rate  
• Instant AI-based screening support  
• Designed for primary healthcare use  
""")

# ===== IMAGE INPUT =====
st.markdown("## 📷 Upload or Capture Image")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload Skin Image", type=["jpg", "jpeg", "png"])

with col2:
    camera_image = st.camera_input("Take Live Photo")

img = None
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
elif camera_image:
    img = Image.open(camera_image).convert("RGB")

# ===== PREDICTION =====
if img:

    st.markdown("---")
    col_img, col_pred = st.columns([1, 2])

    with col_img:
        st.image(img, caption="Selected Image", use_column_width=True)

    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    sorted_idx = preds.argsort()[::-1]

    top_class = CLASSES[sorted_idx[0]]
    top_type = CLASS_INFO[top_class]['type']
    overall_conf = preds[sorted_idx[0]] * 100

    with col_pred:
        st.markdown("## 🥇 Top 3 Predictions")

        top3_idx = sorted_idx[:3]
        cols = st.columns(3)

        for i, idx in enumerate(top3_idx):

            cls = CLASSES[idx]
            conf = preds[idx] * 100
            cls_name = CLASS_INFO[cls]['name']
            cls_type = CLASS_INFO[cls]['type']

            if cls_type == "Cancer":
                color = "#e74c3c"
                bg_color = "#fdecea"
                emoji = "🚨"
            else:
                color = "#2ecc71"
                bg_color = "#eafaf1"
                emoji = "✅"

            with cols[i]:
                st.markdown(
                    f"""
                    <div style="
                        background-color:{bg_color};
                        border:3px solid {color};
                        padding:35px;
                        border-radius:20px;
                        text-align:center;
                        box-shadow:0 6px 18px rgba(0,0,0,0.15);
                        margin-bottom:20px;">
                        <h2>{emoji} {cls_name}</h2>
                        <h1 style="color:{color}; font-size:42px;">
                            {conf:.2f}%
                        </h1>
                        <p style="font-size:20px; color:gray;">
                            Type: {cls_type}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.markdown("---")
        st.markdown("## 🔍 All 7 Class Predictions")

        for idx in sorted_idx:
            cls = CLASSES[idx]
            conf = preds[idx] * 100
            cls_name = CLASS_INFO[cls]['name']
            cls_type = CLASS_INFO[cls]['type']

            if cls_type == "Benign":
                st.success(f"✅ {cls_name} — {conf:.2f}%")
            elif cls_type == "Precancer":
                st.warning(f"⚠️ {cls_name} — {conf:.2f}%")
            else:
                st.error(f"🚨 {cls_name} — {conf:.2f}%")

        st.markdown("## 🚦 Risk Level")

        if top_type == "Benign":
            st.success("🟢 Low Risk (Benign Lesion)")
        elif top_type == "Precancer":
            if overall_conf > 70:
                st.warning("🟡 Medium Risk (Precancerous Lesion)")
            else:
                st.info("🟢 Low to Medium Risk")
        else:
            if overall_conf > 75:
                st.error("🔴 High Risk (Cancer Detected)")
            elif overall_conf > 50:
                st.warning("🟡 Medium Risk (Possible Cancer)")
            else:
                st.info("🟢 Low Confidence Cancer Prediction")

        st.markdown("## 🎯 Model Confidence")
        st.progress(int(overall_conf))
        st.write(f"Confidence Score: {overall_conf:.2f}%")

        st.markdown("## 🩺 Medical Recommendation")

        if top_type == "Benign":
            st.success("Lesion appears benign. Regular monitoring recommended.")
        elif top_type == "Precancer":
            st.warning("Possible precancerous lesion. Consult dermatologist soon.")
        else:
            st.error("High-risk lesion detected. Immediate medical consultation recommended.")

        st.markdown("## 📄 Download Report")

        report = f"""
AI Skin Cancer Screening Report
Date: {datetime.datetime.now()}

Top Prediction: {CLASS_INFO[top_class]['name']}
Type: {top_type}
Confidence: {overall_conf:.2f}%

Disclaimer:
This tool is for screening purposes only.
Consult a certified dermatologist.
"""

        st.download_button(
            label="Download Report",
            data=report,
            file_name="skin_cancer_report.txt",
            mime="text/plain"
        )

st.markdown("---")
st.markdown("## ⚠️ Disclaimer")
st.warning("""
This AI tool is for preliminary screening only.
It is not a replacement for professional medical diagnosis.
Always consult a certified dermatologist.
""")

st.markdown("---")
st.markdown("## ⚙️ How It Works")
st.write("""
1️⃣ User uploads or captures a skin lesion image  
2️⃣ Image resized to 224x224 pixels  
3️⃣ MobileNetV2 extracts deep features  
4️⃣ Softmax layer predicts probability across 7 classes  
5️⃣ Smart risk logic combines class + confidence  
""")

st.markdown("## 📊 Dataset Information")
st.write("""
Dataset Used: HAM10000  
Total Images: 10,015  
Classes: 7 Skin Lesion Categories  
Model: Transfer Learning (MobileNetV2)  
""")

st.markdown("## 🌍 Future Deployment Vision")
st.write("""
• Integration with rural health camps  
• Mobile app for ASHA workers  
• Cloud-based hospital dashboard  
• Tele-dermatology integration  
• Government healthcare partnerships  
""")
