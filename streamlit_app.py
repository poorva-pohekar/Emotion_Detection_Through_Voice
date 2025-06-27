import streamlit as st
import librosa.display
import numpy as np
import librosa
import matplotlib.pyplot as plt
from model_utils import load_model
from data_preprocessing import extract_features
from recorder import record_audio
import time
import os

st.set_page_config(page_title="Emotion Detection from Voice")

# model = load_model()
model, scaler, pca = load_model()


st.title("🎙️ Emotion Detection from Voice")
st.write("Record or upload your voice to detect your emotion.")

# To store path of input audio
audio_path = None

# Record Audio
if st.button("🎤 Record Voice"):
    st.info("Recording... Please speak.")
    start_time = time.time()
    record_audio()
    end_time = time.time()
    duration = round(end_time - start_time, 2)
    audio_path = "input.wav"
    st.success(f"✅ Recording completed. Duration: {duration} seconds")
    st.audio(audio_path, format="audio/wav")

# Upload Audio
uploaded_file = st.file_uploader("📁 Or Upload an Audio File (WAV/MP3)", type=["wav", "mp3"])
if uploaded_file is not None:
    audio_path = "uploaded_audio.wav"
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.read())
    st.audio(audio_path, format="audio/wav")
    st.success("✅ File uploaded successfully!")

# Process and Predict
if audio_path:
    try:
        # features = extract_features(audio_path).reshape(1, -1)
        features = extract_features(audio_path).reshape(1, -1)
        features = scaler.transform(features)
        features = pca.transform(features)
        prediction = model.predict(features)[0]



        # prediction = model.predict(features)[0]
        # st.success(f"🎯 Detected Emotion: **{prediction.upper()}**")
        # Map prediction to emotion name first
        int_to_emotion = {
            0: "neutral",
            1: "calm",
            2: "happy",
            3: "sad",
            4: "angry",
            5: "fearful",
            6: "disgust",
            7: "surprised"
        }

        emotion_label = int_to_emotion.get(prediction, "Unknown")
        st.success(f"🎯 Detected Emotion: **{emotion_label.upper()}**")




        # Optional: Plot Mel Spectrogram
        audio, sr = librosa.load(audio_path)
        mel = librosa.feature.melspectrogram(y=audio, sr=sr)
        fig, ax = plt.subplots()
        img = librosa.display.specshow(librosa.power_to_db(mel, ref=np.max), sr=sr, ax=ax, y_axis='mel', x_axis='time')
        ax.set(title='Mel Spectrogram')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"⚠️ Error processing audio: {e}")
