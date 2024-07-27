import json
import requests
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import random
import threading

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = FastAPI()

class CombinedInput(BaseModel):
    text: str


json_filepath = r'C:\JAS\Infiheal_main\combined_articles_summaries.json'
csv_filepath = r'C:\JAS\Infiheal\mentalhealth (1).csv'
classification_csv_filepath = r'C:\JAS\Infiheal\Combined Data.csv'

def load_data(filepath):
    try:
        with open(filepath, 'r') as file:
            return json.load(file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading data from {filepath}: {e}")

def load_csv_data(filepath):
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading CSV data from {filepath}: {e}")

print("Loading articles data...")
articles = load_data(json_filepath)
print("Articles data loaded.")

print("Loading additional questions data...")
additional_questions_df = load_csv_data(csv_filepath)
print("Additional questions data loaded.")

print("Loading classification data...")
classification_data = load_csv_data(classification_csv_filepath)
classification_data = classification_data.rename(columns={"statement": "text", "status": "label"})
classification_data = classification_data.dropna(subset=["text", "label"])
print("Classification data loaded.")

def preprocess(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()
    filtered_text = [lemmatizer.lemmatize(w) for w in word_tokens if w.isalnum() and w not in stop_words]
    return ' '.join(filtered_text)

classification_data['text'] = classification_data['text'].apply(preprocess)

def train_classification_model(data):
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)
    return model, vectorizer

classification_model, vectorizer = train_classification_model(classification_data)
print("Classification model trained.")

def search_articles(query, articles):
    query_tokens = set(preprocess(query).split())
    results = []
    for article in articles:
        title_tokens = set(preprocess(article['title']).split())
        summary_tokens = set(preprocess(article['summary']).split())
        if query_tokens & title_tokens or query_tokens & summary_tokens:
            results.append(article)
    return results

def search_csv_questions(query, df):
    query_tokens = set(preprocess(query).split())
    for index, row in df.iterrows():
        question_tokens = set(preprocess(row['Questions']).split())
        if query_tokens & question_tokens:
            return row['Answers'], row.get('url', None)
    return None, None

def construct_prompt(query, articles):
    if articles:
        article_info = ". ".join([f"{article['title']} (Link: {article.get('url', 'No URL available')})" for article in articles[:3]])
        prompt = f"Based on your interest in mental health topics like '{query}', you might find these articles insightful: {article_info}."
    else:
        prompt = f"I couldn't find specific articles related to '{query}', but let's discuss your concerns about mental health."
    prompt += " How can I assist you further today?"
    return prompt

def query_model(prompt, model_id, api_token):
    API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {api_token}"}
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    if response.status_code == 200:
        return response.json()
    else:
        raise HTTPException(status_code=response.status_code, detail=f"Model {model_id} failed with error: {response.text}")

def query_hf_with_fallback(prompt):
    api_token = "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"  # Replace with your actual API token
    models = ["bigscience/bloom", "gpt2", "EleutherAI/gpt-neo-2.7B"]

    for model_id in models:
        try:
            response = query_model(prompt, model_id, api_token)
            return response
        except HTTPException as e:
            print(e.detail)
            continue

    return {"error": "All models failed"}

def get_empathy_statement():
    statements = [
        "It's okay to feel this way. Let's see how we can help.",
        "I'm sorry to hear that. You are not alone in this.",
        "I understand that this can be tough. There are ways to cope.",
        "Let's explore some ways to manage these feelings.",
        "I am here to support you. Let's find some solutions together."
    ]
    return random.choice(statements)

@app.post("/process")
def process_endpoint(input: CombinedInput):
    query = input.text
    found_articles = search_articles(query, articles)
    csv_answer, csv_url = search_csv_questions(query, additional_questions_df)
    
    response = {"classification": None, "rag": None}
    
    # Classification part
    try:
        processed_text = vectorizer.transform([preprocess(query)])
        prediction = classification_model.predict(processed_text)
        response["classification"] = prediction[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making classification prediction: {e}")

    # RAG part
    greeting = "Hi there! "
    empathy = get_empathy_statement()
    
    if csv_answer:
        rag_response = f"{greeting}{empathy} {csv_answer}"
        if csv_url:
            rag_response += f" You can read more about it here: {csv_url}"
    else:
        prompt = construct_prompt(query, found_articles)
        rag_response = query_hf_with_fallback(prompt)
        if isinstance(rag_response, list) and rag_response and 'generated_text' in rag_response[0]:
            rag_response = rag_response[0]['generated_text']
        elif isinstance(rag_response, dict) and 'generated_text' in rag_response:
            rag_response = rag_response['generated_text']
        else:
            rag_response = {"error": "Unexpected response structure", "details": rag_response}

    if found_articles:
        articles_info = "\n".join([f"- [{article['title']}]({article['url']})" for article in found_articles[:3]])
        response["rag"] = f"{rag_response}\n\nRelated Articles:\n{articles_info}"
    else:
        response["rag"] = rag_response

    return response

# Function to start FastAPI server in a thread
def start_server():
    uvicorn.run(app, host="127.0.0.1", port=8000)

# Function to interact with the user in the terminal
def user_interaction():
    print("Welcome to the interactive terminal. Type 'exit' to stop.")
    while True:
        user_input = input("Enter a query or classification text: ").strip()
        if user_input.lower() == 'exit':
            break

        response = requests.post("http://127.0.0.1:8000/process", json={"text": user_input})
        
        if response.status_code == 200:
            print("Response:", response.json())
        else:
            print("Error:", response.status_code, response.json())

# Start the FastAPI server in a separate thread
server_thread = threading.Thread(target=start_server)
server_thread.start()

# Start user interaction in the main thread
user_interaction()

# Ensure the server thread is joined properly before exiting
server_thread.join()
