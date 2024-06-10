import os
import re
import string
import numpy as np
from flask import Flask, request, jsonify, render_template
from nltk.stem import SnowballStemmer
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Inicialización de Flask
app = Flask(__name__, static_url_path='/static')

# Path de los datos
data_path = r'D:\U\7. Septimo\RI\ProyectoRI\data\training_txt'
stopwords_path = r'D:\U\7. Septimo\RI\ProyectoRI\data\stopwords.txt'

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

# Obtener los nombres de las características del vectorizador BoW
feature_names_bow = vectorizer_bow.get_feature_names_out()

# Construir el índice invertido para BoW
inverted_index_bow = build_inverted_index_from_bow(X_bow, feature_names_bow)

# Función para encontrar documentos relevantes
def relevant_documents_for_query(query_terms, index):
    relevant_docs = set()
    for term in query_terms:
        if term in index:
            relevant_docs.update(index[term])
    return relevant_docs

# Función para calcular precisión y recall
def precision_recall(ranked_documents, relevant_docs):
    retrieved_docs = set(ranked_documents)
    TP = len(retrieved_docs & relevant_docs)
    FP = len(retrieved_docs - relevant_docs)
    FN = len(relevant_docs - retrieved_docs)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    return precision, recall

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('q', '')
    query_processed = preprocess_text(query)
    query_vector_bow = vectorizer_bow.transform([query_processed])
    query_vector_tfidf = vectorizer_tfidf.transform([query_processed])

    similarity_scores_bow = cosine_similarity(query_vector_bow, X_bow)
    similarity_scores_tfidf = cosine_similarity(query_vector_tfidf, X_tfidf)

    ranked_documents_bow = np.argsort(similarity_scores_bow[0])[::-1]
    ranked_documents_tfidf = np.argsort(similarity_scores_tfidf[0])[::-1]

    # Convertir índices a nombres de archivos
    ranked_files_bow = [filenames[idx] for idx in ranked_documents_bow[:10]]
    ranked_files_tfidf = [filenames[idx] for idx in ranked_documents_tfidf[:10]]

    results = {
        "BoW": ranked_files_bow,
        "TF-IDF": ranked_files_tfidf
    }

    # Imprimir la respuesta para depuración
    print(results)

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
