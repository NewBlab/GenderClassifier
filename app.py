"""
Streamlit App: 🎙️ Gender Classifier (Enhanced UI)

Dependencies:
    pip install streamlit streamlit-audiorecorder librosa numpy soundfile

Run:
    streamlit run app.py
"""
import streamlit as st
from audiorecorder import audiorecorder
import io
import numpy as np
import soundfile as sf
import librosa

# ─────────────── Page Config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎙️ Gender Classifier Based on Pitch",
    page_icon="🎤",
    layout="wide",
)

# ─────────────── Custom CSS ──────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .reportview-container {
            background-color: #f9f9fc;
        }
        h1 {
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .stButton>button {
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎙️ Gender Classifier Based on Pitch")

# ─────────────── Sidebar Settings ────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    default_threshold = st.session_state.get("threshold", 165)
    threshold = st.slider(
        "Pitch Threshold (Hz)",
        min_value=50,
        max_value=300,
        value=default_threshold,
        step=1,
        help="Mean pitch above this value will be classified as **Female**; otherwise **Male**.",
    )
    st.session_state["threshold"] = threshold
    st.markdown(f"**Current threshold:** {threshold} Hz")
    st.divider()
    st.markdown(
        "ℹ️ Adjust the threshold if predictions seem inaccurate for your microphone or voice."
    )

# ─────────────── Tabs Layout ─────────────────────────────────────────────────────
tab_record, tab_results, tab_about = st.tabs(["🎤 Record", "📊 Results", "❓ About"])

# ─────────────── Record Tab ──────────────────────────────────────────────────────
with tab_record:
    st.subheader("Step 1 — Record your voice")
    st.caption(
        """
        • Click **Record** and speak naturally for about 5 seconds.  
        • Click **Stop** and wait for the waveform preview to appear.  
        • Once your recording shows up, proceed to the **Results** tab.
        """
    )
    wav_bytes = audiorecorder("🔴 Record / Stop", key="gender_sample")
    if wav_bytes:
        with st.expander("▶️ Play back your recording"):
            wav_io = io.BytesIO()
            wav_bytes.export(wav_io, format="wav")
            st.audio(wav_io.getvalue(), format="audio/wav")
        st.success("Recording captured! Head to the **Results** tab ➡️")

# ─────────────── Results Tab ─────────────────────────────────────────────────────
with tab_results:
    st.subheader("Step 2 — View results")
    if "gender_sample" not in st.session_state or not wav_bytes:
        st.info("No recording detected. Please record a sample first.")
    else:
        # Convert AudioSegment to WAV bytes
        wav_io = io.BytesIO()
        wav_bytes.export(wav_io, format="wav")
        wav_io.seek(0)

        # Read audio data
        data, fs = sf.read(wav_io)

        # Estimate mean pitch
        try:
            f0 = librosa.yin(data, fmin=50, fmax=500, sr=fs)
            mean_pitch = float(np.nanmean(f0))
        except Exception:
            mean_pitch = float("nan")

        if np.isnan(mean_pitch):
            st.error("⚠️ Could not detect pitch. Please try recording again in a quieter environment.")
        else:
            gender = "Female" if mean_pitch > threshold else "Male"

            col1, col2 = st.columns(2)
            col1.metric("Mean Pitch (Hz)", f"{mean_pitch:.1f}")
            col2.metric("Predicted Gender", gender)

            st.markdown(f"*Threshold used:* **{threshold} Hz**")

            if st.button("🔄 Record Another Sample"):
                st.session_state.pop("gender_sample", None)
                st.experimental_rerun()

# ─────────────── About Tab ───────────────────────────────────────────────────────
with tab_about:
    st.subheader("How it Works")
    st.markdown(
        """
        This demo uses **librosa**'s YIN algorithm to estimate the fundamental frequency
        (pitch) of your voice sample.

        1. **Record a sample** using the built‑in audio component.  
        2. The **mean pitch** of the recording is calculated.  
        3. Pitch is compared against a configurable **threshold** (default 165 Hz).  
        4. Above the threshold → **Female**  
        5. Below the threshold → **Male**

        *Tip:* If the prediction seems off, experiment with the threshold slider in
        the sidebar or ensure you record in a quiet environment with the microphone close
        to your mouth.
        """
    )
    st.markdown(
        "<div style='text-align:center; margin-top:1rem;'>Ensure microphone access is enabled in your browser.</div>",
        unsafe_allow_html=True,
    )
