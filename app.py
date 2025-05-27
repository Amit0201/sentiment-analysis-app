import streamlit as st
import joblib
import os
import time
import pandas as pd
from streamlit_webrtc import webrtc_streamer
import speech_recognition as sr
import threading
import queue

MODEL_FILE = os.path.join("models", "sentiment_model.pkl")
VECTORIZER_FILE = os.path.join("models", "vectorizer.pkl")

if not os.path.isfile(MODEL_FILE) or not os.path.isfile(VECTORIZER_FILE):
    st.error(
        f"Model or vectorizer file not found!\n"
        f"Make sure '{MODEL_FILE}' and '{VECTORIZER_FILE}' exist."
    )
    st.stop()

try:
    with open(MODEL_FILE, "rb") as f:
        model = joblib.load(f)
    with open(VECTORIZER_FILE, "rb") as f:
        vectorizer = joblib.load(f)
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

st.set_page_config(page_title="Sentiment Analysis", page_icon="📝")

# CSS Styling + animations (keep your previous CSS, shortened here)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');
    .stApp { background-color: #2c2f4a; font-family: 'Montserrat', sans-serif; color: #f0f0f0; }
    .title { color: #f9a825; font-size: 56px; font-weight: 700; margin-bottom: 30px; text-align: center; text-shadow: 2px 2px 4px rgba(0,0,0,0.7);}
    textarea[aria-label="Your Review"] { background-color: #3a3f6f !important; color: #f0f0f0 !important; border-radius: 10px; padding: 12px !important; font-size: 18px !important; font-family: 'Montserrat', sans-serif !important; border: none !important; box-shadow: 0 0 10px rgba(249,168,37,0.6); resize: vertical !important; min-height: 150px !important; transition: box-shadow 0.3s ease;}
    textarea[aria-label="Your Review"]:focus { box-shadow: 0 0 20px #f9a825; outline: none !important;}
    div.stButton > button { background: linear-gradient(90deg, #f9a825, #fbc02d); color: #2c2f4a; font-weight: 700; font-size: 20px; padding: 12px 30px; border-radius: 25px; border: none; box-shadow: 0 4px 8px rgba(249,168,37,0.6); cursor: pointer; margin-top: 10px;}
    div.stButton > button:hover { background: linear-gradient(90deg, #fbc02d, #f9a825);}
    footer { text-align: center; font-size: 14px; color: #999; margin-top: 40px; opacity: 0; animation: fadeIn 3s forwards 1.5s;}
    @keyframes fadeIn { to {opacity: 1;}}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<h1 class="title">Sentiment Analysis</h1>', unsafe_allow_html=True)

# Initialize session state for review history
if "history" not in st.session_state:
    st.session_state.history = []

# Function to predict sentiment and return sentiment label and confidence
def predict_sentiment(text):
    vect = vectorizer.transform([text])
    pred = model.predict(vect)[0]
    prob = model.predict_proba(vect)[0]
    confidence = max(prob)
    sentiment = "Positive 😊" if pred == 1 else "Negative 😠"
    return sentiment, confidence

# Voice input implementation
def voice_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Please speak now.")
        audio = r.listen(source, phrase_time_limit=5)
    try:
        text = r.recognize_google(audio)
        st.success("Transcribed text:")
        st.write(text)
        return text
    except sr.UnknownValueError:
        st.error("Could not understand audio")
        return ""
    except sr.RequestError as e:
        st.error(f"Could not request results; {e}")
        return ""

# Alternative: Using streamlit-webrtc for live voice input (simple button approach)
voice_mode = st.checkbox("Use voice input instead of typing")

if voice_mode:
    st.warning("Click 'Record' button and allow microphone access. Speak clearly for up to 5 seconds.")
    if st.button("Record"):
        try:
            text = voice_input()
            if text:
                review = text
            else:
                review = ""
        except Exception as e:
            st.error(f"Error with voice input: {e}")
            review = ""
    else:
        review = ""
else:
    review = st.text_area("Your Review", height=150)

threshold = st.slider(
    "Confidence threshold to show result:",
    min_value=0.5,
    max_value=1.0,
    value=0.75,
    step=0.01,
    help="Adjust the minimum confidence required to display prediction."
)

if st.button("Analyze Sentiment"):
    if not review.strip():
        st.warning("⚠️ Please enter a review to analyze.")
    else:
        sentiment, confidence = predict_sentiment(review)
        if confidence < threshold:
            st.info(f"Prediction confidence ({confidence:.2f}) is below threshold ({threshold}). Try entering a clearer review.")
        else:
            st.markdown(f"### Prediction: **{sentiment}**")
            st.markdown(f"**Confidence:** {confidence * 100:.2f}%")
            progress_bar = st.progress(0)
            for i in range(int(confidence * 100) + 1):
                progress_bar.progress(i)
                time.sleep(0.01)
            if sentiment.startswith("Positive"):
                st.balloons()

            # Save to history
            st.session_state.history.append(
                {"Review": review, "Sentiment": sentiment, "Confidence": f"{confidence*100:.2f}%"}
            )

# Show sentiment history in expandable section
with st.expander("Show Sentiment History"):
    if st.session_state.history:
        df_history = pd.DataFrame(st.session_state.history)
        st.dataframe(df_history, use_container_width=True)

        # Download button to export history as CSV
        csv = df_history.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download History as CSV",
            data=csv,
            file_name="sentiment_history.csv",
            mime="text/csv",
        )
    else:
        st.info("No sentiment history yet. Analyze some reviews!")

# Footer credit
st.markdown('<footer>Made with ❤️ using Streamlit & Python | by AMIT DIXIT.</footer>', unsafe_allow_html=True)
