import streamlit as st
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

import os
port = int(os.environ.get("PORT", 8501))

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="TruthLens — AI Detection",
    page_icon="🔬",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
    --ink:      #0d0f14;
    --ink2:     #13161e;
    --ink3:     #1a1f2e;
    --ink4:     #242938;
    --lime:     #c8ff57;
    --violet:   #7c3aed;
    --violet2:  #a78bfa;
    --rose:     #fb7185;
    --sky:      #38bdf8;
    --text:     #f1f5f9;
    --sub:      #94a3b8;
    --glass:    rgba(255,255,255,0.04);
    --gborder:  rgba(255,255,255,0.08);
    --r:        16px;
}

*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: var(--ink) !important;
    color: var(--text) !important;
    font-family: 'Outfit', sans-serif !important;
}

/* ── animated noise grain overlay ── */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
    opacity: 0.6;
}

/* ── glowing orbs in background ── */
[data-testid="stAppViewContainer"]::after {
    content: '';
    position: fixed;
    top: -200px; left: -200px;
    width: 600px; height: 600px;
    background: radial-gradient(circle, rgba(124,58,237,0.12) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

[data-testid="block-container"] {
    position: relative;
    z-index: 1;
    padding: 2rem 3rem !important;
    max-width: 1200px !important;
}

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: var(--ink2) !important;
    border-right: 1px solid var(--gborder) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 2rem;
}

/* ── typography ── */
h1 {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 900 !important;
    font-size: 3rem !important;
    line-height: 1.1 !important;
    letter-spacing: -2px !important;
    color: var(--text) !important;
    margin-bottom: 0 !important;
}

h2, h3, h4 {
    font-family: 'Outfit', sans-serif !important;
    color: var(--text) !important;
}

p, li, span {
    font-family: 'Outfit', sans-serif !important;
}

/* ── tabs ── */
[data-testid="stTabs"] {
    margin-top: 2rem;
}

[data-testid="stTabs"] [role="tablist"] {
    background: transparent !important;
    border-bottom: 1px solid var(--gborder) !important;
    gap: 0 !important;
    padding: 0 !important;
}

[data-testid="stTabs"] [role="tab"] {
    background: transparent !important;
    color: var(--sub) !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.2px !important;
    padding: 12px 28px !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    transition: all 0.2s !important;
    text-transform: none !important;
    margin-bottom: -1px !important;
}

[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--lime) !important;
    border-bottom-color: var(--lime) !important;
    background: transparent !important;
}

[data-testid="stTabs"] [role="tab"]:hover:not([aria-selected="true"]) {
    color: var(--text) !important;
}

[data-testid="stTabsContent"] {
    background: transparent !important;
    border: none !important;
    padding: 28px 0 !important;
}

/* ── inputs ── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background: var(--ink3) !important;
    border: 1px solid var(--gborder) !important;
    border-radius: var(--r) !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    padding: 14px 18px !important;
    transition: all 0.25s !important;
    caret-color: var(--lime) !important;
}

[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--lime) !important;
    box-shadow: 0 0 0 3px rgba(200,255,87,0.08), inset 0 1px 0 rgba(255,255,255,0.05) !important;
    outline: none !important;
}

[data-testid="stTextInput"] input::placeholder,
[data-testid="stTextArea"] textarea::placeholder {
    color: var(--sub) !important;
    opacity: 0.5 !important;
}

[data-testid="stTextInput"] label,
[data-testid="stTextArea"] label {
    color: var(--sub) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    margin-bottom: 6px !important;
}

/* ── buttons ── */
[data-testid="stButton"] button {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    border-radius: 999px !important;
    padding: 9px 22px !important;
    transition: all 0.2s cubic-bezier(0.34,1.56,0.64,1) !important;
    letter-spacing: 0.1px !important;
    position: relative !important;
    overflow: hidden !important;
}

/* example buttons */
[data-testid="stButton"] button {
    background: var(--ink3) !important;
    border: 1px solid var(--gborder) !important;
    color: var(--sub) !important;
}

[data-testid="stButton"] button:hover {
    background: var(--ink4) !important;
    border-color: rgba(255,255,255,0.15) !important;
    color: var(--text) !important;
    transform: translateY(-2px) scale(1.02) !important;
}

[data-testid="stButton"] button:active {
    transform: scale(0.97) !important;
}

/* analyze button — the one with Analyze in the text */
button[kind="primary"],
[data-testid="stButton"]:has(button p:contains("Analyze")) button,
[data-testid="stButton"]:has(button div:contains("Analyze")) button {
    background: var(--lime) !important;
    border: none !important;
    color: #0d0f14 !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 20px rgba(200,255,87,0.25), 0 1px 0 rgba(255,255,255,0.2) inset !important;
}

button[kind="primary"]:hover {
    box-shadow: 0 8px 32px rgba(200,255,87,0.4) !important;
    transform: translateY(-3px) scale(1.03) !important;
    color: #0d0f14 !important;
}

/* ── file uploader ── */
[data-testid="stFileUploader"] {
    background: var(--ink3) !important;
    border: 2px dashed var(--gborder) !important;
    border-radius: var(--r) !important;
    transition: all 0.25s !important;
    padding: 8px !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(200,255,87,0.3) !important;
    background: var(--ink4) !important;
}

[data-testid="stFileUploader"] * {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    color: var(--sub) !important;
}

[data-testid="stFileUploader"] label {
    color: var(--sub) !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    font-size: 0.7rem !important;
}

/* ── alerts ── */
[data-testid="stSuccess"] {
    background: rgba(74,222,128,0.06) !important;
    border: 1px solid rgba(74,222,128,0.2) !important;
    border-left: 3px solid #4ade80 !important;
    border-radius: var(--r) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    color: #4ade80 !important;
    padding: 16px 20px !important;
}

[data-testid="stError"] {
    background: rgba(251,113,133,0.06) !important;
    border: 1px solid rgba(251,113,133,0.2) !important;
    border-left: 3px solid var(--rose) !important;
    border-radius: var(--r) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    color: var(--rose) !important;
    padding: 16px 20px !important;
}

[data-testid="stInfo"] {
    background: rgba(56,189,248,0.05) !important;
    border: 1px solid rgba(56,189,248,0.15) !important;
    border-left: 3px solid var(--sky) !important;
    border-radius: var(--r) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    color: var(--sky) !important;
    padding: 14px 20px !important;
}

[data-testid="stWarning"] {
    background: rgba(251,191,36,0.06) !important;
    border: 1px solid rgba(251,191,36,0.2) !important;
    border-left: 3px solid #fbbf24 !important;
    border-radius: var(--r) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    color: #fbbf24 !important;
}

/* ── image ── */
[data-testid="stImage"] img {
    border-radius: var(--r) !important;
    border: 1px solid var(--gborder) !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4) !important;
}

/* ── spinner ── */
[data-testid="stSpinner"] > div {
    border-top-color: var(--lime) !important;
}

/* ── hr ── */
hr {
    border: none !important;
    border-top: 1px solid var(--gborder) !important;
    margin: 2rem 0 !important;
}

/* ── scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--ink); }
::-webkit-scrollbar-thumb { background: var(--ink4); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: rgba(200,255,87,0.3); }

/* ── sidebar text ── */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] span {
    font-family: 'Outfit', sans-serif !important;
    color: var(--text) !important;
}

/* ── column gaps ── */
[data-testid="stHorizontalBlock"] {
    gap: 16px !important;
}

/* ── subheader ── */
[data-testid="stHeadingWithActionElements"] h2 {
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    color: var(--text) !important;
    letter-spacing: -0.3px !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Example Text Data
# -----------------------------
example_fake_title = "Government secretly installs mind control chips in citizens"
example_fake_article = """A viral claim circulating on social media suggests that the government has
installed microchips in citizens through vaccines in order to control thoughts.
Experts and officials have strongly denied the claim and stated there is no
scientific evidence supporting this theory."""

example_real_title = "Scientists discover new method to improve solar panel efficiency"
example_real_article = """Researchers from a leading university have developed a new material that
significantly increases solar panel efficiency. The technology could make
renewable energy more affordable and accelerate the transition to clean power."""

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_text_model():
    model = joblib.load("models/text/svm_model.pkl")
    vectorizer = joblib.load("models/text/tfidf_vectorizer.pkl")
    return model, vectorizer

@st.cache_resource
def load_image_model():
    return tf.keras.models.load_model("models/image/cnn_model.keras")

text_model, vectorizer = load_text_model()
image_model = load_image_model()
IMG_SIZE = (128, 128)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.markdown("""
<div style='padding:0 4px'>
  <div style='font-family:Outfit,sans-serif;font-size:1.4rem;font-weight:900;letter-spacing:-1px;margin-bottom:4px'>
    🔬 TruthLens
  </div>
  <div style='font-family:JetBrains Mono,monospace;font-size:0.68rem;color:#64748b;letter-spacing:1px;text-transform:uppercase;margin-bottom:20px'>
    AI Misinformation Detector
  </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style='background:#1a1f2e;border:1px solid rgba(255,255,255,0.06);border-radius:14px;padding:16px;margin-bottom:12px'>
  <div style='font-family:JetBrains Mono,monospace;font-size:0.65rem;color:#64748b;letter-spacing:2px;text-transform:uppercase;margin-bottom:10px'>Text Pipeline</div>
  <div style='display:flex;flex-wrap:wrap;gap:6px'>
    <span style='background:rgba(200,255,87,0.08);border:1px solid rgba(200,255,87,0.2);border-radius:6px;font-family:JetBrains Mono,monospace;font-size:0.7rem;color:#c8ff57;padding:4px 10px'>TF-IDF</span>
    <span style='background:rgba(200,255,87,0.08);border:1px solid rgba(200,255,87,0.2);border-radius:6px;font-family:JetBrains Mono,monospace;font-size:0.7rem;color:#c8ff57;padding:4px 10px'>Linear SVM</span>
  </div>
</div>

<div style='background:#1a1f2e;border:1px solid rgba(255,255,255,0.06);border-radius:14px;padding:16px;margin-bottom:12px'>
  <div style='font-family:JetBrains Mono,monospace;font-size:0.65rem;color:#64748b;letter-spacing:2px;text-transform:uppercase;margin-bottom:10px'>Image Pipeline</div>
  <div style='display:flex;flex-wrap:wrap;gap:6px'>
    <span style='background:rgba(167,139,250,0.08);border:1px solid rgba(167,139,250,0.2);border-radius:6px;font-family:JetBrains Mono,monospace;font-size:0.7rem;color:#a78bfa;padding:4px 10px'>Custom CNN</span>
    <span style='background:rgba(167,139,250,0.08);border:1px solid rgba(167,139,250,0.2);border-radius:6px;font-family:JetBrains Mono,monospace;font-size:0.7rem;color:#a78bfa;padding:4px 10px'>Batch Norm</span>
    <span style='background:rgba(167,139,250,0.08);border:1px solid rgba(167,139,250,0.2);border-radius:6px;font-family:JetBrains Mono,monospace;font-size:0.7rem;color:#a78bfa;padding:4px 10px'>Augmentation</span>
  </div>
</div>

<div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px'>
  <div style='background:#1a1f2e;border:1px solid rgba(255,255,255,0.06);border-radius:14px;padding:14px;text-align:center'>
    <div style='font-family:Outfit,sans-serif;font-size:1.6rem;font-weight:800;color:#c8ff57;line-height:1'>44K</div>
    <div style='font-family:JetBrains Mono,monospace;font-size:0.6rem;color:#64748b;letter-spacing:1px;text-transform:uppercase;margin-top:4px'>Articles</div>
  </div>
  <div style='background:#1a1f2e;border:1px solid rgba(255,255,255,0.06);border-radius:14px;padding:14px;text-align:center'>
    <div style='font-family:Outfit,sans-serif;font-size:1.6rem;font-weight:800;color:#a78bfa;line-height:1'>2K</div>
    <div style='font-family:JetBrains Mono,monospace;font-size:0.6rem;color:#64748b;letter-spacing:1px;text-transform:uppercase;margin-top:4px'>Images</div>
  </div>
</div>

<div style='background:rgba(200,255,87,0.04);border:1px solid rgba(200,255,87,0.1);border-radius:10px;padding:10px 14px;margin-top:8px'>
  <div style='font-family:JetBrains Mono,monospace;font-size:0.65rem;color:#94a3b8;line-height:1.6'>
    SVM Accuracy<br>
    <span style='color:#c8ff57;font-size:1rem;font-weight:700'>99.55%</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style='margin-top:16px;font-family:JetBrains Mono,monospace;font-size:0.62rem;color:#334155;text-align:center;padding-top:12px;border-top:1px solid rgba(255,255,255,0.05)'>
Final Year Research Project — 2026
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Hero Header
# -----------------------------
st.markdown("""
<div style='margin-bottom:2rem'>
  <div style='display:flex;align-items:center;gap:14px;margin-bottom:10px'>
    <div style='width:44px;height:44px;background:linear-gradient(135deg,#c8ff57,#7c3aed);border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:1.4rem;flex-shrink:0'>🔬</div>
    <div>
      <div style='font-family:Outfit,sans-serif;font-size:2.2rem;font-weight:900;letter-spacing:-1.5px;line-height:1;color:#f1f5f9'>TruthLens</div>
      <div style='font-family:JetBrains Mono,monospace;font-size:0.7rem;color:#64748b;letter-spacing:2px;text-transform:uppercase;margin-top:2px'>Hybrid AI Detection System</div>
    </div>
  </div>
  <div style='display:flex;gap:10px;flex-wrap:wrap;margin-top:16px'>
    <span style='background:rgba(200,255,87,0.08);border:1px solid rgba(200,255,87,0.15);border-radius:999px;font-family:JetBrains Mono,monospace;font-size:0.7rem;color:#c8ff57;padding:5px 14px'>✦ SVM text classifier</span>
    <span style='background:rgba(167,139,250,0.08);border:1px solid rgba(167,139,250,0.15);border-radius:999px;font-family:JetBrains Mono,monospace;font-size:0.7rem;color:#a78bfa;padding:5px 14px'>✦ CNN image detector</span>
    <span style='background:rgba(56,189,248,0.08);border:1px solid rgba(56,189,248,0.15);border-radius:999px;font-family:JetBrains Mono,monospace;font-size:0.7rem;color:#38bdf8;padding:5px 14px'>✦ No login required</span>
    <span style='background:rgba(251,113,133,0.08);border:1px solid rgba(251,113,133,0.15);border-radius:999px;font-family:JetBrains Mono,monospace;font-size:0.7rem;color:#fb7185;padding:5px 14px'>✦ Real-time prediction</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2 = st.tabs(["📰  Text — Fake News Detection", "🖼  Image — Deepfake Detection"])

# =====================================================
# TEXT DETECTION
# =====================================================
with tab1:
    st.markdown("""
    <div style='background:linear-gradient(135deg,rgba(200,255,87,0.05) 0%,rgba(124,58,237,0.05) 100%);border:1px solid rgba(200,255,87,0.1);border-radius:16px;padding:20px 24px;margin-bottom:24px'>
      <div style='font-family:Outfit,sans-serif;font-weight:700;font-size:1.1rem;color:#f1f5f9;margin-bottom:6px'>Analyze News Article</div>
      <div style='font-family:JetBrains Mono,monospace;font-size:0.72rem;color:#64748b;line-height:1.6'>
        Paste a news title and article body. The TF-IDF + Linear SVM pipeline will classify it as <span style='color:#c8ff57'>REAL</span> or <span style='color:#fb7185'>FAKE</span>.
      </div>
    </div>
    """, unsafe_allow_html=True)

    colA, colB = st.columns(2)
    with colA:
        if st.button("⚠  Load Example Fake News", use_container_width=True):
            st.session_state["title"] = example_fake_title
            st.session_state["article"] = example_fake_article
    with colB:
        if st.button("✅  Load Example Real News", use_container_width=True):
            st.session_state["title"] = example_real_title
            st.session_state["article"] = example_real_article

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    title = st.text_input(
        "News Title",
        value=st.session_state.get("title", ""),
        placeholder="e.g. Scientists discover cure for common cold…"
    )
    article = st.text_area(
        "Article Body",
        height=180,
        value=st.session_state.get("article", ""),
        placeholder="Paste the full article content here…"
    )

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    if st.button("🔍  Analyze News", use_container_width=False, type="primary"):
        if title.strip() == "" or article.strip() == "":
            st.warning("⚠  Please enter both title and article text.")
        else:
            with st.spinner("Running TF-IDF + SVM pipeline…"):
                text   = title + " " + article
                vector = vectorizer.transform([text])
                prediction = text_model.predict(vector)[0]
                score      = text_model.decision_function(vector)[0]
                confidence = min(abs(score) * 100, 99.9)

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            if prediction == 1:
                st.success("✅  REAL NEWS — Content appears legitimate")
            else:
                st.error("❌  FAKE NEWS — Misinformation detected")
            st.info(f"Confidence Score: {confidence:.2f}%")

# =====================================================
# IMAGE DETECTION
# =====================================================
with tab2:
    st.markdown("""
    <div style='background:linear-gradient(135deg,rgba(167,139,250,0.05) 0%,rgba(251,113,133,0.05) 100%);border:1px solid rgba(167,139,250,0.12);border-radius:16px;padding:20px 24px;margin-bottom:24px'>
      <div style='font-family:Outfit,sans-serif;font-weight:700;font-size:1.1rem;color:#f1f5f9;margin-bottom:6px'>Detect Deepfake Image</div>
      <div style='font-family:JetBrains Mono,monospace;font-size:0.72rem;color:#64748b;line-height:1.6'>
        Upload a face image (JPG / PNG). The CNN model analyzes spatial features to classify it as <span style='color:#4ade80'>REAL</span> or <span style='color:#fb7185'>DEEPFAKE</span>.
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📷  Load Example Real Image", use_container_width=True):
            example_img = Image.open("examples/real.jpg")
            st.session_state["example_image"] = example_img
    with col2:
        if st.button("📷  Load Example Fake Image", use_container_width=True):
            example_img = Image.open("examples/fake.jpg")
            st.session_state["example_image"] = example_img

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        img = Image.open(uploaded_file)
    elif "example_image" in st.session_state:
        img = st.session_state["example_image"]
    else:
        img = None

    if img:
        colA, colB = st.columns([1, 2])
        with colA:
            st.image(img, caption="Input image", use_container_width=True)
        with colB:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            if st.button("🔍  Analyze Image", use_container_width=False, type="primary"):
                with st.spinner("Running CNN inference…"):
                    img_resized = img.resize(IMG_SIZE)
                    img_array  = image.img_to_array(img_resized)
                    img_array  = img_array / 255.0
                    img_array  = np.expand_dims(img_array, axis=0)
                    prediction = image_model.predict(img_array)
                    probability = prediction[0][0]

                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                if probability > 0.5:
                    st.success("✅  REAL IMAGE — No manipulation detected")
                    st.info(f"Confidence: {probability * 100:.2f}%")
                else:
                    st.error("❌  DEEPFAKE — AI-generated face detected")
                    st.info(f"Confidence: {(1 - probability) * 100:.2f}%")

# -----------------------------
# Footer
# -----------------------------
st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='border-top:1px solid rgba(255,255,255,0.06);padding-top:20px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px'>
  <div style='font-family:Outfit,sans-serif;font-size:0.95rem;font-weight:700;color:#334155;letter-spacing:-0.3px'>TruthLens</div>
  <div style='font-family:JetBrains Mono,monospace;font-size:0.65rem;color:#1e293b;letter-spacing:1px;text-transform:uppercase'>
    Hybrid Fake News &amp; Deepfake Detection &nbsp;·&nbsp; Final Year Research Project – 2026
  </div>
</div>
""", unsafe_allow_html=True)