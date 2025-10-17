Live demo:https://sentiment-analysis-of-tweets-111111111.streamlit.app/
soon
# AI-Powered Sentiment Analyzer for Tweets

[span_0](start_span)This project presents an AI-powered sentiment analyzer that integrates concepts from Data Science, Machine Learning, and AI[span_0](end_span). [span_1](start_span)It processes tweets or short text inputs and predicts their sentiment as **Positive**, **Negative**, or **Neutral**, complete with a confidence score[span_1](end_span).

[span_2](start_span)The system is built using Python and deployed as an interactive Streamlit web application[span_2](end_span). [span_3](start_span)The classifier model is a Support Vector Machine (SVM) trained using TF-IDF vectorization[span_3](end_span).

---

## ✨ Live Demo

You can explore the live demo deployed on Streamlit Cloud:

**[span_4](start_span)[https://sentiment-analysis-of-tweets-111111111.streamlit.app/](https://sentiment-analysis-of-tweets-111111111.streamlit.app/)**[span_4](end_span)

---

## 🚀 Features

* **[span_5](start_span)Sentiment Classification**: Predicts if a text input is Positive, Negative, or Neutral[span_5](end_span).
* **[span_6](start_span)Confidence Score**: Displays the model's confidence in its prediction[span_6](end_span).
* **[span_7](start_span)[span_8](start_span)Interactive UI**: A Streamlit interface simulates an intelligent assistant for real-time analysis[span_7](end_span)[span_8](end_span).
* **[span_9](start_span)Visual Feedback**: Uses dynamic visuals and emoji representations for enhanced user engagement[span_9](end_span).
* **[span_10](start_span)Real-time Analysis**: Can analyze user-provided text instantly[span_10](end_span).

---

## 🛠️ Technology Stack

The main technologies used in this project are:

* **[span_11](start_span)Python**[span_11](end_span)
* **[span_12](start_span)Streamlit**: For the interactive web deployment[span_12](end_span).
* **[span_13](start_span)Scikit-learn**: For TF-IDF vectorization and the Support Vector Machine (SVM) classification model[span_13](end_span).
* **[span_14](start_span)NLTK**: For natural language preprocessing, including tokenization and lemmatization[span_14](end_span).
* **[span_15](start_span)Joblib**: For saving and loading the trained ML models[span_15](end_span).
* **[span_16](start_span)Libraries (`re`, `emoji`)**: Used for advanced data cleaning and preprocessing[span_16](end_span).

---

## 🔧 Methodology

The project was executed in three main stages:

1.  **Data Science (Preprocessing)**
    * [span_17](start_span)Tweets were loaded and cleaned using Python libraries like `re`, `nltk`, and `emoji`[span_17](end_span).
    * [span_18](start_span)Cleaning steps included removing URLs, mentions, special characters, and stopwords[span_18](end_span).
    * [span_19](start_span)Advanced steps like lemmatization and negation handling were applied to improve linguistic accuracy[span_19](end_span).
    * [span_20](start_span)Features were extracted using `TfidfVectorizer` with n-grams (up to trigrams) and a large feature set[span_20](end_span).

2.  **Machine Learning (Modeling)**
    * [span_21](start_span)The cleaned text was converted into feature vectors using TF-IDF[span_21](end_span).
    * [span_22](start_span)A linear Support Vector Machine (SVM) model was trained for the classification task[span_22](end_span).
    * [span_23](start_span)The model was evaluated using metrics like accuracy, precision, recall, and F1-score[span_23](end_span).

3.  **Artificial Intelligence (Deployment)**
    * [span_24](start_span)The trained model and vectorizer were integrated into a Streamlit web application[span_24](end_span).
    * [span_25](start_span)The application interface was designed to be interactive, simulating an intelligent assistant that visualizes sentiment in real-time[span_25](end_span).

---

## 📊 Model Performance

[span_26](start_span)[span_27](start_span)The trained SVM model achieved an overall **accuracy of 71%** on the test set[span_26](end_span)[span_27](end_span).

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

*[span_28](start_span)(Based on the classification report in the project document[span_28](end_span))*

---

## 📁 Project Files

* [span_29](start_span)`deploy.py`: The main Streamlit deployment file[span_29](end_span).
* [span_30](start_span)`project.ipynb`: The Jupyter notebook containing the full workflow for data preprocessing, feature extraction, and model training[span_30](end_span).
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
