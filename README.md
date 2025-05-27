# 🎬 Sentiment Analysis on Movie Reviews
A web app that classifies movie reviews into positive or negative sentiments using a machine learning model trained on movie review data. Built with Streamlit for an interactive user experience, including text input and speech recognition features, and natural language processing.

## 🔧 Tech Stack
- Python
- Scikit-learn
- Streamlit
- TF-IDF Vectorization
- Logistic Regression

## 📖 Project Description
This app uses a pre-trained sentiment analysis model based on a logistic regression classifier. It analyzes movie reviews entered by the user and predicts whether the sentiment is positive or negative. The app also displays the prediction confidence and provides a clean, user-friendly interface.
Key features:
- Text input for review analysis
- Speech-to-text input for hands-free review entry
- Real-time prediction with confidence scores
- Interactive and attractive UI with animations

## 🧠 Model Performance
- Accuracy on test set: ~85%
- Vectorizer: TF-IDF (max_df=0.7, stop_words='english')

## 🚀 Run Locally
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## 🪢 Stremlit app link
https://sentiment-analysis-app-b7pqwbnseb58xbqjw2x2ez.streamlit.app/
