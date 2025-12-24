import gradio as gr
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import string

# --- 1. Load the TechHub Data ---
def load_data():
    try:
        # quotechar='"' allows us to handle the complex sentences in your new CSV
        df = pd.read_csv('faq_data.csv', quotechar='"')
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return pd.DataFrame(columns=["question", "answer"])

df = load_data()

# --- 2. Train the "Brain" ---
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = str(text).lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

if not df.empty:
    vectorizer = TfidfVectorizer(preprocessor=preprocess)
    tfidf_matrix = vectorizer.fit_transform(df['question'])
else:
    print("⚠️ Error: faq_data.csv is empty or missing!")

# --- 3. The Response Logic ---
def get_response(user_input, history):
    if df.empty:
        return "System Error: Database not loaded."
    
    if not user_input:
        return ""

    user_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vec, tfidf_matrix)
    
    best_idx = similarities.argmax()
    score = similarities[0, best_idx]
    
    # If confidence is low, give a support number
    if score < 0.4:
        return "I'm not sure about that. Please contact our support team at 1800-TECH-HUB."
    else:
        return df.iloc[best_idx]['answer']

# --- 4. The Interface (Updated with New Examples) ---
custom_css = """
.container {max-width: 800px; margin: auto; padding-top: 20px;}
footer {visibility: hidden}
"""

demo = gr.ChatInterface(
    fn=get_response,
    title="🛒 TechHub AI Assistant",
    description="Welcome to TechHub! Ask me about **Orders**, **Warranties**, **EMI Options**, or **Returns**.",
    theme=gr.themes.Soft(primary_hue="cyan", neutral_hue="slate"),
    css=custom_css,
    # UPDATED EXAMPLES TO MATCH YOUR NEW CSV
    examples=[
        "Do you have EMI options?", 
        "Is there a warranty on laptops?", 
        "How do I return a product?", 
        "Do you ship internationally?",
        "Where is my order?"
    ],
    retry_btn=None,
    undo_btn=None,
    clear_btn="🗑️ Clear Chat"
)

if __name__ == "__main__":
    demo.launch()