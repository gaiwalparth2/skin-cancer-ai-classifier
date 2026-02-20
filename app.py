import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import datetime
import gdown
import os
import sqlite3
import hashlib

# ===== PAGE CONFIG =====
st.set_page_config(page_title="🩺 Skin Cancer AI Classifier", layout="wide")

# ==============================
# 🔐 PROFESSIONAL AUTH SYSTEM
# ==============================

# ----- DATABASE -----
conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS users(
    username TEXT PRIMARY KEY,
    password TEXT
)
""")
conn.commit()

# ----- HASH PASSWORD -----
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ----- REGISTER USER -----
def register_user(username, password):
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                  (username, hash_password(password)))
        conn.commit()
        return True
    except:
        return False

# ----- LOGIN USER -----
def login_user(username, password):
    c.execute("SELECT * FROM users WHERE username=? AND password=?",
              (username, hash_password(password)))
    return c.fetchone()

# ----- SESSION -----
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None

# ----- AUTH PAGE -----
def auth_page():

    st.markdown("""
    <h2 style='text-align:center;'>🔐 Skin Cancer AI Classifier</h2>
    """, unsafe_allow_html=True)

    menu = ["Login", "Register"]
    choice = st.radio("Select Option", menu)

    if choice == "Register":
        st.subheader("📝 Create New Account")
        new_user = st.text_input("Username")
        new_pass = st.text_input("Password", type="password")

        if st.button("Register"):
            if register_user(new_user, new_pass):
                st.success("Account Created Successfully ✅")
                st.info("Now go to Login tab")
            else:
                st.error("Username already exists ❌")

    if choice == "Login":
        st.subheader("🔐 Login to Continue")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            result = login_user(username, password)
            if result:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"Welcome {username} ✅")
                st.rerun()
            else:
                st.error("Invalid Username or Password ❌")

# ----- STOP IF NOT LOGGED IN -----
if not st.session_state.logged_in:
    auth_page()
    st.stop()

# ----- SIDEBAR USER INFO -----
st.sidebar.markdown(f"👋 Logged in as: **{st.session_state.username}**")
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.username = None
    st.rerun()

# ==============================
# 🧠 ORIGINAL AI APP STARTS
# ==============================

# ===== CONFIG =====
IMG_SIZE = 224
MODEL_PATH = "best_skin_cancer_model.h5"

CLASS_INFO = {
    'akiec': {'name': 'Actinic Keratoses', 'type': 'Precancer'},
    'bcc': {'name': 'Basal Cell Carcinoma', 'type': 'Cancer'},
    'bkl': {'name': 'Benign Keratosis', 'type': 'Benign'},
    'df': {'name': 'Dermatofibroma', 'type': 'Benign'},
    'mel': {'name': 'Melanoma', 'type': 'Cancer'},
    'nv': {'name': 'Melanocytic Nevi', 'type': 'Benign'},
    'vasc': {'name': 'Vascular Lesion', 'type': 'Benign'}
}

CLASSES = list(CLASS_INFO.keys())

# ===== LOAD MODEL =====
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(
            "https://drive.google.com/uc?id=17ZZS2rvhKarloodrh_3Gh0v_FQFmGF_n",
            MODEL_PATH,
            quiet=False
        )
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ===== SIDEBAR =====
st.sidebar.title("📊 Model Overview")
st.sidebar.metric("Model", "MobileNetV2")
st.sidebar.metric("Accuracy", "91%")
st.sidebar.metric("Validation Accuracy", "88%")
st.sidebar.metric("Classes", "7")
st.sidebar.markdown("---")
st.sidebar.write("National Healthcare Hackathon 🇮🇳")

# ===== HEADER =====
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

# ===== REST OF YOUR ORIGINAL CODE CONTINUES SAME =====

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
