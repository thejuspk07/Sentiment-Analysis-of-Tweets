# Sentiment Analysis of Tweets

[![Live Demo](https://img.shields.io/badge/Demo-Live-green?style=flat-square)](https://sentiment-analysis-of-tweets-111111111.streamlit.app/)

A machine learning project for performing sentiment analysis on tweets using a Support Vector Machine (SVM) classifier and TF-IDF vectorization. This project enables users to analyze the sentiment (positive or negative) of tweets and visualize the results in an interactive web application built with Streamlit.

## 🚀 Live Demo

Experience the application here:  
👉 [Sentiment Analysis of Tweets – Live Demo](https://sentiment-analysis-of-tweets-111111111.streamlit.app/)

---

## 📚 Project Overview

This repository contains:::

- Data preprocessing and cleaning pipeline for tweets
- Feature extraction using TF-IDF vectorizer
- Sentiment classification model using SVM
- Pre-trained model and vectorizer for fast inference
- Streamlit app for interactive sentiment analysis

## ✨ Features

- Clean and preprocess tweet text data
- Train and evaluate an SVM classifier
- Predict sentiment of new/unseen tweets
- User-friendly web interface for real-time sentiment analysis
- Visualizations of sentiment predictions...
- predicts

## 🗂️ Repository Structure

```
.
├── Tweets.csv                  # Dataset of tweets and sentiments
├── project.ipynb               # Data analysis and model training notebook
├── deploy.py                   # Streamlit app source code
├── svm_sentiment_model.pkl     # Trained SVM model
├── tfidf_vectorizer.pkl        # Trained TF-IDF vectorizer
├── requirements.txt            # Python dependencies
├── models/                     # Directory for additional models (if any)
└── README.md                   # Project documentation
```

## 🛠️ Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/thejuspk07/Sentiment-Analysis-of-Tweets.git
   cd Sentiment-Analysis-of-Tweets
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app locally:**
   ```bash
   streamlit run deploy.py
   ```

## 📈 Usage

- Open the [live demo](https://sentiment-analysis-of-tweets-111111111.streamlit.app/) or run locally as above.
- Enter your tweet in the web interface.
- Click "Analyze" to see the predicted sentiment and visualization.techinique

## 📦 Dependencies

- Python ≥ 3.7
- pandas
- numpy
- scikit-learn
- streamlit

See `requirements.txt` for the full list.

## 📝 Dataset

The project uses the `Tweets.csv` file which contains labeled tweets for training and evaluation. Please ensure you have the necessary licenses to use this dataset.

## 🤖 Model Files

- `svm_sentiment_model.pkl` – Pre-trained SVM model for sentiment classification
- `tfidf_vectorizer.pkl` – Fitted TF-IDF vectorizer

These files are used by the deployment app for fast inference.

## 📒 Notebooks

- `project.ipynb` – Contains the steps for data preprocessing, feature extraction, training, and evaluation.

## 📤 Deployment

The Streamlit app can be deployed on [Streamlit Cloud](https://streamlit.io/cloud) or similar platforms. Modify `deploy.py` as needed to customize the UI or add new features.

## 🙏 Acknowledgements

- The dataset and inspiration are based on public sentiment analysis challenges.
- Thanks to the open-source ML and data science community!

## 📬 Contact

For questions or feedback, open an issue or contact [thejuspk07](https://github.com/thejuspk07).

-----

[Back to top](#sentiment-analysis-of-tweets)
