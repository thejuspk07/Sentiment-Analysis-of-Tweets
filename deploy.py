import streamlit as st
import joblib
import emoji
import re
import nltk
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="😊",
    layout="centered"
)

# Enhanced AI-themed styling with animations
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Helvetica Neue", sans-serif;
}


/* ======== Background ======== */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    animation: gradientShift 20s ease infinite;
    background-size: 300% 300%;
    color: #f1f5f9;
}
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ======== Main Card ======== */

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(30px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ======== Title & Subtitle ======== */
h1 {
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    font-weight: 700;
    font-size: 3rem;
    margin-bottom: 10px;
    animation: titlePulse 3s ease-in-out infinite;
}
@keyframes titlePulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.8; }
}
.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 1.1rem;
    margin-bottom: 25px;
}

/* ======== Sentiment Result Boxes ======== */
.sentiment-box {
    padding: 30px;
    border-radius: 20px;
    margin: 20px 0;
    text-align: center;
    backdrop-filter: blur(12px);
    animation: scaleIn 0.5s cubic-bezier(0.68, -0.55, 0.265, 1.55);
    position: relative;
    overflow: hidden;
    color: white;
}
@keyframes scaleIn {
    from { opacity: 0; transform: scale(0.85); }
    to { opacity: 1; transform: scale(1); }
}
.sentiment-box::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: linear-gradient(45deg, transparent, rgba(255,255,255,0.2), transparent);
    animation: shimmer 3s infinite;
}
@keyframes shimmer {
    0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
    100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
}

/* Distinct Sentiment Colors */
.positive {
    background: linear-gradient(135deg, #16a34a 0%, #22c55e 100%);
    border: 2px solid rgba(187, 247, 208, 0.7);
    box-shadow: 0 0 50px rgba(34, 197, 94, 0.5);
}
.negative {
    background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
    border: 2px solid rgba(254, 202, 202, 0.7);
    box-shadow: 0 0 50px rgba(239, 68, 68, 0.5);
}
.neutral {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    border: 2px solid rgba(191, 219, 254, 0.7);
    box-shadow: 0 0 50px rgba(59, 130, 246, 0.5);
}

.sentiment-box h2 {
    color: #fff;
    font-size: 2.4rem;
    margin-bottom: 10px;
    text-shadow: 0 2px 15px rgba(0, 0, 0, 0.3);
}
.sentiment-box p {
    color: #f9fafb;
    font-size: 1.2rem;
    font-weight: 500;
}

/* ======== Buttons ======== */
.stButton > button {
    width: 100%;
    margin-top: 10px;
    background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 50%, #6366f1 100%);
    color: #ffffff;
    border: none;
    padding: 15px 30px;
    border-radius: 15px;
    font-weight: 600;
    font-size: 1.1rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(59, 130, 246, 0.4);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #6366f1 0%, #3b82f6 50%, #06b6d4 100%);
    box-shadow: 0 6px 25px rgba(14, 165, 233, 0.6);
    transform: translateY(-2px);
}
.stButton > button:active {
    transform: translateY(0);
}



/* ======== Text Area ======== */
.stTextArea > div > div > textarea {
    border-radius: 15px;
    border: 2px solid rgba(255, 255, 255, 0.3);
    background: rgba(255, 255, 255, 0.05);
    color: #f1f5f9;
    padding: 15px;
    font-size: 1rem;
    transition: all 0.3s ease;
}
.stTextArea > div > div > textarea:focus {
    border-color: #38bdf8;
    box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.2);
    transform: scale(1.01);
}

/* ======== Confidence Bar ======== */
.confidence-bar {
    width: 100%;
    height: 8px;
    background: rgba(255, 255, 255, 0.2);
    border-radius: 10px;
    margin-top: 10px;
}
.confidence-fill {
    height: 100%;
    background: linear-gradient(90deg, #fef08a, #fde047);
    border-radius: 10px;
    animation: fillBar 1s ease-out;
}
@keyframes fillBar { from { width: 0; } }

/* ======== Expander, Alerts, and Footer ======== */
.stExpander {
    border-radius: 15px;
    border: 1px solid rgba(255, 255, 255, 0.15);
    background: rgba(255, 255, 255, 0.08);
}
.stAlert {
    border-radius: 15px;
    color: #f8fafc;
    background: rgba(239, 68, 68, 0.2);
}
hr {
    border: none;
    height: 2px;
    background: linear-gradient(90deg, transparent, #38bdf8, transparent);
    margin: 40px 0;
}
.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 0.9rem;
}

/* ======== Emoji Animation ======== */
.emoji-float {
    font-size: 4rem;
    animation: float 3s ease-in-out infinite;
}
@keyframes float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-20px); }
}
</style>
""", unsafe_allow_html=True)


# Initialize NLTK resources
@st.cache_resource
def initialize_nltk():
    try:
        for resource in ['wordnet', 'omw-1.4', 'stopwords']:
            try:
                nltk.data.find(f'corpora/{resource}')
            except LookupError:
                nltk.download(resource, quiet=True)
        
        from nltk.corpus import wordnet
        from nltk.corpus import stopwords
        _ = wordnet.VERB
        _ = stopwords.words('english')
        return True
    except Exception as e:
        logger.error(f"Error loading NLTK resources: {str(e)}")
        return False

# Load models
@st.cache_resource
def load_models():
    try:
        model_path = Path("models/ensemble_model.pkl")
        vectorizer_path = Path("models/vectorizer.pkl")
        
        if not model_path.parent.exists():
            raise FileNotFoundError(f"Models directory not found at {model_path.parent.absolute()}")
            
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path.absolute()}")
        if not vectorizer_path.exists():
            raise FileNotFoundError(f"Vectorizer file not found at {vectorizer_path.absolute()}")
            
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        logger.info("Model and vectorizer loaded successfully")
        return model, vectorizer
        
    except FileNotFoundError as e:
        logger.error(f"Model files not found: {str(e)}")
        st.error(f"⚠️ {str(e)}\nPlease run the training notebook to generate the model files.")
        return None, None
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        st.error(f"⚠️ Error loading models: {str(e)}")
        return None, None

# Initialize resources
nltk_initialized = initialize_nltk()
if not nltk_initialized:
    st.error("⚠️ Failed to initialize NLTK resources. Please try restarting the application.")
    st.stop()

try:
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    model, vectorizer = load_models()
except Exception as e:
    st.error(f"⚠️ Error initializing text processing tools: {str(e)}")
    st.stop()

def clean_tweet_advanced(tweet):
    """Clean and preprocess tweet text for sentiment analysis."""
    if not isinstance(tweet, (str, bytes)):
        tweet = str(tweet)
    
    try:
        tweet = tweet.lower().strip()
        tweet = re.sub(r'http\S+|www\S+|@\S+', '', tweet)
        tweet = re.sub(r'[^A-Za-z0-9\s!?]', '', tweet)
        
        # Handle emoji conversion
        try:
            tweet = emoji.demojize(tweet)
        except Exception as emoji_err:
            logger.warning(f"Emoji processing failed: {str(emoji_err)}")
            # Continue processing even if emoji conversion fails
        
        tweet = re.sub(r'\bnot\b (\w+)', r'not_\1', tweet)
        
        words = []
        for w in tweet.split():
            if w not in stop_words:
                try:
                    lemmatized = lemmatizer.lemmatize(w)
                    if lemmatized:
                        words.append(lemmatized)
                except Exception as lem_err:
                    logger.warning(f"Lemmatization failed for word '{w}': {str(lem_err)}")
                    words.append(w)  # Use original word if lemmatization fails
        
        cleaned_text = " ".join(words)
        if not cleaned_text.strip():
            logger.warning("Text became empty after cleaning")
            return "neutral"
        return cleaned_text
        
    except Exception as e:
        logger.error(f"Critical error cleaning tweet: {str(e)}")
        return "neutral"  # Return a default value instead of raising to prevent crashes

def analyze_sentiment(text, confidence_threshold=0.7):
    """Analyze sentiment of input text and return prediction with confidence."""
    if not text or not text.strip():
        logger.warning("Empty text provided for sentiment analysis")
        return None, None, None, "Text is empty"

    try:
        clean_text = clean_tweet_advanced(text)
        if clean_text == "neutral":  # Check for the default return from clean_tweet_advanced
            logger.info("Text was cleaned to neutral state")
            return "neutral", 1.0, None, None

        if not model or not vectorizer:
            logger.error("Model or vectorizer not initialized")
            return None, None, None, "Model not initialized"
            
        features = vectorizer.transform([clean_text])
        prediction = model.predict(features)[0]
        confidence = 1.0  # Default confidence
        
        # Get confidence scores if available
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(features)[0]
                confidence = float(max(probabilities))
            except Exception as e:
                logger.warning(f"Error getting probability estimates: {str(e)}")
        elif hasattr(model, 'decision_function'):
            try:
                decision_scores = model.decision_function(features)
                confidence = 1 / (1 + np.exp(-np.abs(decision_scores).max()))
            except Exception as e:
                logger.warning(f"Error getting decision function scores: {str(e)}")
        else:
            logger.warning("Model doesn't support probability estimates or decision scores")
        
        return prediction, confidence, clean_text, None

    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        return None, None, None, str(e)

# Page title
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.title("✨ Sentiment Analyzer")
st.markdown('<p class="subtitle">Discover the emotion behind your words with AI-powered analysis</p>', unsafe_allow_html=True)

# Initialize session state
if 'text_value' not in st.session_state:
    st.session_state.text_value = ""
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False

import random

def random_choice():
    A='I absolutely love this product! It’s fantastic and exceeded my expectations. Highly recommended! 😊'
    B='how is your day today'
    C='The product is okay. Nothing special, but it does what it’s supposed to do. 😐'
    D='The service was outstanding, and the staff went above and beyond to help me. I’m really impressed! 🌟'
    E='I’m really disappointed with this experience. Nothing worked as promised, and support didn’t respond. 😠' 
    F='It’s an average experience. Not too bad, but nothing remarkable either. 🤷‍♂️'
    G = "Absolutely fantastic! Everything worked perfectly, and I couldn’t be happier. 😍"
    H = "What an amazing experience! Totally worth it and exceeded all my expectations. 🌈"
    I = "I’m so satisfied with my purchase. Great quality and fast delivery! 👍"
    J = "Loved it! The design, the feel, everything was just perfect. 💖"
    K = "Superb quality and excellent value for money. Would definitely buy again! 🥰"
    L = "Terrible experience. The product broke after just one use. 😤"
    M = "I regret buying this. It’s nothing like what was advertised. 😩"
    N = "Poor quality and bad customer support. Definitely not worth the price. 👎"
    O = "Completely disappointed. I expected much better for the cost. 😔"
    P = "It stopped working within a week. Such a waste of money. 😡"
    Q = "My day’s been decent."
    R = "It’s okay, nothing special."
    S = "Just another normal day."
    T = "Feeling alright."
    U = "It’s fine, I guess."
    V = "Nothing much happening."
    W = "I’m doing okay."
    X = "Pretty average day."
    Y = "Everything’s normal."
    Z = "Can’t complain, it’s okay."


    options = [A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z]
    choice = random.choice(options)
    return choice

def set_example():
    st.session_state.text_input = random_choice()
# Text input
user_input = st.text_area(
    "Enter your text to analyze:",
    placeholder="Type or paste your text here... ✍️",
    height=150,
    key="text_input"
)

# Button columns for better layout
col1, col2 = st.columns(2)

with col1:
    analyze_btn = st.button("🔍 Analyze Sentiment", use_container_width=True)

with col2:
    example_btn = st.button("✨ Try Example", on_click=set_example, use_container_width=True)


# Analysis
if analyze_btn:
    if user_input and user_input.strip():
        with st.spinner("🤖 Analyzing your text..."):
            prediction, confidence, clean_text, error = analyze_sentiment(user_input)
            
            if error:
                st.error(f"🚫 Error: {error}")
            else:
                # Display sentiment with enhanced visuals
                emoji_map = {
                    "positive": "😊",
                    "negative": "😔",
                    "neutral": "😐"
                }
                
                sentiment_emoji = emoji_map.get(prediction.lower(), '🤔')
                
                st.markdown(f"""
                    <div class="sentiment-box {prediction.lower()}">
                        <div class="emoji-float">{sentiment_emoji}</div>
                        <h2>{prediction.upper()}</h2>
                        <p>Confidence: {confidence:.1%}</p>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: {confidence * 100}%"></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                st.session_state.analyzed = True
                
                # Show analysis details
                with st.expander("📊 View Analysis Details"):
                    st.markdown("**Original Text:**")
                    st.info(user_input)
                    st.markdown("**Processed Text:**")
                    st.success(clean_text)
                    st.markdown("**Confidence Score:**")
                    st.progress(confidence)
    else:
        st.warning("⚠️ Please enter some text to analyze.")

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
