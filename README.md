# AI-Powered Sentiment Analyzer for Tweets

This project presents an AI-powered sentiment analyzer that integrates concepts from Data Science, Machine Learning, and AI. It processes tweets or short text inputs and predicts their sentiment as **Positive**, **Negative**, or **Neutral**, complete with a confidence score.

The system is built using Python and deployed as an interactive Streamlit web application. The classifier model is a Support Vector Machine (SVM) trained using TF-IDF vectorization.

---

## ✨ Live Demo

You can explore the live demo deployed on Streamlit Cloud:

**[https://sentiment-analysis-of-tweets-111111111.streamlit.app/](https://sentiment-analysis-of-tweets-111111111.streamlit.app/)**

---

## 🚀 Features

* **Sentiment Classification**: Predicts if a text input is Positive, Negative, or Neutral.
* **Confidence Score**: Displays the model's confidence in its prediction.
* **Interactive UI**: A Streamlit interface simulates an intelligent assistant for real-time analysis.
* **Visual Feedback**: Uses dynamic visuals and emoji representations for enhanced user engagement.
* **Real-time Analysis**: Can analyze user-provided text instantly.

---

## 🛠️ Technology Stack

The main technologies used in this project are:

* **Python**
* **Streamlit**: For the interactive web deployment.
* **Scikit-learn**: For TF-IDF vectorization and the Support Vector Machine (SVM) classification model.
* **NLTK**: For natural language preprocessing, including tokenization and lemmatization.
* **Joblib**: For saving and loading the trained ML models.
* **Libraries (`re`, `emoji`)**: Used for advanced data cleaning and preprocessing.

---

## 🔧 Methodology

The project was executed in three main stages:

1.  **Data Science (Preprocessing)**
    * Tweets were loaded and cleaned using Python libraries like `re`, `nltk`, and `emoji`.
    * Cleaning steps included removing URLs, mentions, special characters, and stopwords.
    * Advanced steps like lemmatization and negation handling were applied to improve linguistic accuracy.
    * Features were extracted using `TfidfVectorizer`.

2.  **Machine Learning (Modeling)**
    * The cleaned text was converted into feature vectors using TF-IDF.
    * A linear Support Vector Machine (SVM) model was trained for the classification task.
    * The model was evaluated using metrics like accuracy, precision, recall, and F1-score.

3.  **Artificial Intelligence (Deployment)**
    * The trained model and vectorizer were integrated into a Streamlit web application.
    * The application interface was designed to be interactive, simulating an intelligent assistant that visualizes sentiment in real-time.

---

## 📊 Model Performance

The trained SVM model achieved an overall **accuracy of 71%** on the test set.

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Negative** | 0.75 | 0.58 | 0.66 | 1572 |
| **Neutral** | 0.64 | 0.78 | 0.70 | 2236 |
| **Positive** | 0.80 | 0.73 | 0.76 | 1688 |
| | | | | |
| **Accuracy** | | | **0.71** | **5496** |
| **Macro Avg** | 0.73 | 0.70 | 0.71 | 5496 |
| **Weighted Avg**| 0.72 | 0.71 | 0.71 | 5496 |

*(Based on the classification report in the project document)*

---

## 📁 Project Files

* `deploy.py`: The main Streamlit deployment file.
* `project.ipynb`: The Jupyter notebook containing the full workflow for data preprocessing, feature extraction, and model training.
* *(Inferred)* `model.joblib`: The saved/serialized SVM model.
* *(Inferred)* `vectorizer.joblib`: The saved/serialized TF-IDF vectorizer.

---

## 💻 How to Run Locally

To run this project on your local machine:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository.git](https://github.com/your-username/your-repository.git)
    cd your-repository
    ```

2.  **Install the dependencies:**
    ```bash
    pip install streamlit scikit-learn nltk emoji joblib pandas
    ```

3.  **Download NLTK data:**
    Run the following in a Python interpreter:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    ```

4.  **Run the Streamlit application:**
    (Ensure the `deploy.py` file and the saved `.joblib` model/vectorizer files are in the root directory).
    ```bash
    streamlit run deploy.py
    ```
