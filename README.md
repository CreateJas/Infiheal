About the project

This project is a FastAPI-based mental health chatbot designed to provide empathetic responses and relevant article suggestions based on user queries. The chatbot uses machine learning models for classification and retrieval-augmented generation (RAG) techniques to enhance the user experience.

Technologies Used and Their Roles
FastAPI

Role: FastAPI is used to create the web server and define the endpoints for the chatbot.
Function: It handles HTTP requests, processes input data, and returns responses to users.
Uvicorn

Role: Uvicorn is an ASGI web server used to run the FastAPI application.
Function: It serves the FastAPI application, allowing it to handle multiple requests asynchronously and efficiently.
NLTK (Natural Language Toolkit)

Role: NLTK is used for natural language processing (NLP) tasks.
Function: It provides tools for tokenization, stopword removal, and lemmatization to preprocess the text data.
Scikit-learn

Role: Scikit-learn is used for building the machine learning model for text classification.
Function: It includes tools for vectorizing text data (TF-IDF), splitting the dataset, training the Logistic Regression model, and making predictions.
Pandas

Role: Pandas is used for data manipulation and analysis.
Function: It loads and processes CSV data, handles dataframes, and performs data cleaning and preprocessing.
Requests

Role: Requests library is used to make HTTP requests to external APIs.
Function: It sends POST requests to the Hugging Face API to query language models for generating responses.
Approach
Data Loading and Preprocessing:

Article data is loaded from a JSON file, and additional questions data and classification data are loaded from CSV files.
Text data is preprocessed using NLTK to remove stopwords, tokenize the text, and apply lemmatization.
Text Classification:

The classification data is split into training and testing sets.
TF-IDF vectorization is applied to the text data to convert it into numerical features.
A Logistic Regression model is trained on the TF-IDF features to classify the text into predefined categories.
Article Search:

User queries are preprocessed, and relevant articles are searched based on matching tokens in the article titles and summaries.
Additional questions data is searched to find direct answers to user queries.
Constructing Responses:

Empathetic statements are generated to provide supportive responses to user queries.
Relevant articles are included in the response if found.
If no direct answers or articles are found, a prompt is constructed, and the Hugging Face API is queried to generate a fallback response using language models like BLOOM, GPT-2, or GPT-Neo.
FastAPI Endpoint:

A /process endpoint is defined to handle user queries.
The endpoint processes the input, performs classification, searches for relevant articles and answers, and constructs a response combining these elements.
Model Querying and Fallback
Query Model:
The constructed prompt is sent to the Hugging Face API to generate responses using various models like BLOOM, GPT-2, or GPT-Neo.
The models are queried in sequence, and a response is returned if successful. If all models fail, an error message is generated.
Summary
The mental health chatbot leverages natural language processing, machine learning classification, and retrieval-augmented generation to provide relevant and empathetic responses to user queries. The integration of various technologies ensures efficient handling of requests and accurate responses based on the user's input.
