# 🎬 Sentiment Analysis on Movie Reviews

This is a simple web app that predicts whether a movie review is positive or negative using Natural Language Processing.

## 🔧 Tech Stack
- Python
- Scikit-learn
- Streamlit
- TF-IDF Vectorization
- Logistic Regression

## 🧠 Model Performance
- Accuracy on test set: ~85%
- Vectorizer: TF-IDF (max_df=0.7, stop_words='english')

## 🚀 Run Locally
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
