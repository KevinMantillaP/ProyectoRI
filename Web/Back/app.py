import os
import re
import string
import numpy as np
from flask import Flask, request, jsonify, render_template
from nltk.stem import SnowballStemmer
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity 
from nltk.metrics.distance import jaccard_distance

# Inicialización de Flask
app = Flask(__name__, static_url_path='/static')

# Path de los datos
data_path = r'C:\Users\kevin\OneDrive\Documentos\GitHub\ProyectoRI\data\training_txt'
stopwords_path = r'C:\Users\kevin\OneDrive\Documentos\GitHub\ProyectoRI\data\stopwords.txt'

# Cargar documentos
documents = []
for filename in os.listdir(data_path):
    if filename.endswith('.txt'):
        path = os.path.join(data_path, filename)
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read()
            documents.append((filename, content))

# Cargar stopwords
with open(stopwords_path, 'r', encoding='utf-8') as file:
    stop_words = set(file.read().splitlines())

# Inicializar el stemmer
stemmer = SnowballStemmer('english')

# Función de preprocesamiento
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    processed_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(processed_tokens)

# Preprocesar los documentos
preprocessed_documents = [(filename, preprocess_text(content)) for filename, content in documents]

# Extraer los contenidos preprocesados
filenames, contents = zip(*preprocessed_documents)

# Vectorización BoW
vectorizer_bow = CountVectorizer()
X_bow = vectorizer_bow.fit_transform(contents)

# Vectorización TF-IDF
vectorizer_tfidf = TfidfVectorizer()
X_tfidf = vectorizer_tfidf.fit_transform(contents)

# Construir índice invertido para BoW
def build_inverted_index_from_bow(X, feature_names):
    inverted_index = {}
    for term_idx, term in enumerate(feature_names):
        term_docs = set(X[:, term_idx].nonzero()[0])
        inverted_index[term] = term_docs
    return inverted_index

# Construir índice invertido para TF-IDF
def build_inverted_index_from_tfidf(X, feature_names):
    inverted_index = {}
    for doc_idx, doc in enumerate(X):
        doc = doc.toarray().flatten()
        relevant_terms = set(np.where(doc > 0)[0])
        for term_idx in relevant_terms:
            term = feature_names[term_idx]
            if term not in inverted_index:
                inverted_index[term] = set()
            inverted_index[term].add(doc_idx)
    return inverted_index

# Obtener los nombres de las características del vectorizador BoW y TF-IDF
feature_names_bow = vectorizer_bow.get_feature_names_out()
feature_names_tfidf = vectorizer_tfidf.get_feature_names_out()

# Construir el índice invertido para BoW y TF-IDF
inverted_index_bow = build_inverted_index_from_bow(X_bow, feature_names_bow)
inverted_index_tfidf = build_inverted_index_from_tfidf(X_tfidf, feature_names_tfidf)

# Función para encontrar documentos relevantes usando Jaccard distance
def relevant_documents_for_query_jaccard(query_terms, index):
    relevant_docs = set()
    for term in query_terms:
        if term in index:
            relevant_docs.update(index[term])
    return relevant_docs

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('q', '')
    query_processed = preprocess_text(query)

    # Calcular similitud con Jaccard para BoW
    query_terms = set(query_processed.split())
    relevant_docs_bow = relevant_documents_for_query_jaccard(query_terms, inverted_index_bow)

    # Calcular similitud de Jaccard y ordenar documentos relevantes para BoW
    similarities_bow = []
    for idx, content in enumerate(contents):
        doc_terms = set(content.split())
        similarity = 1 - jaccard_distance(query_terms, doc_terms)
        similarities_bow.append((idx, similarity))

    similarities_bow.sort(key=lambda x: x[1], reverse=True)
    ranked_documents_bow = [filenames[idx] for idx, _ in similarities_bow[:10]]
    # Calcular similitud de coseno y ordenar documentos relevantes para TF-IDF
    query_vector_tfidf = vectorizer_tfidf.transform([query_processed])
    similarities_tfidf = cosine_similarity(query_vector_tfidf, X_tfidf).flatten()
    similarities_tfidf_indices = np.argsort(similarities_tfidf)[::-1]
    ranked_documents_tfidf = [filenames[idx] for idx in similarities_tfidf_indices[:10]]

    results = {
        "BoW": ranked_documents_bow,
        "TF-IDF": ranked_documents_tfidf,
    }

    # Imprimir la respuesta para depuración
    print(results)

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
