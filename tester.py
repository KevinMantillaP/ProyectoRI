import os
import re
import string
import numpy as np
from flask import Flask, request, jsonify, render_template
from nltk.stem import SnowballStemmer
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Path de los datos
data_path = r'C:\Users\usuario\Fer-Pc\Escritorio\EPN\2024-A\SEPTIMO_SEMESTRE\RECUPERACION_DE_INFORMACION\repoMantillaRI\ProyectoRI\data\training_txt'
stopwords_path = r'C:\Users\usuario\Fer-Pc\Escritorio\EPN\2024-A\SEPTIMO_SEMESTRE\RECUPERACION_DE_INFORMACION\repoMantillaRI\ProyectoRI\data\stopwords.txt'

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

preprocessed_contents = [content for _, content in preprocessed_documents]
document_indices = [index for index, _ in preprocessed_documents]

# Paso 3: Vectorización BoW
vectorizer_bow = CountVectorizer()
X_bow = vectorizer_bow.fit_transform(preprocessed_contents)

# Paso 4: Definir funciones para el cálculo de la similitud de Jaccard
def binary_vector(vector):
    """Convierte un vector a binario."""
    return (vector > 0).astype(int)

def jaccard_similarity(vector1, vector2):
    """Calcula la similitud de Jaccard entre dos vectores binarios."""
    intersection = np.sum(vector1 & vector2)
    union = np.sum(vector1 | vector2)
    return intersection / union if union != 0 else 0

# Paso 5: Definir el motor de búsqueda
class SearchEngine:
    def __init__(self, vectorizer, document_vectors, documents, indices):
        self.vectorizer = vectorizer
        self.document_vectors = document_vectors
        self.documents = documents
        self.indices = indices
    
    def search(self, query):
        query_vector = self.vectorizer.transform([query]).toarray()
        bin_query_vector = binary_vector(query_vector)
        
        scores = []
        for i, doc_vector in enumerate(self.document_vectors):
            bin_doc_vector = binary_vector(doc_vector.toarray())
            similarity = jaccard_similarity(bin_query_vector, bin_doc_vector)
            scores.append((similarity, self.indices[i]))
        
        scores.sort(reverse=True, key=lambda x: x[0])
        return scores[:10]  # Retornar solo el top 10

# Crear instancia del motor de búsqueda
documents = [doc for doc in preprocessed_contents]
indices = [index for index, _ in preprocessed_documents]
engine = SearchEngine(vectorizer_bow, X_bow, documents, indices)


def leer_documentos_relevantes(archivo):
    documentos_relevantes = {}
    with open(archivo, 'r') as file:
        for line in file:
            partes = line.strip().split()
            if len(partes) != 4:  # Asegurarse de que hay 4 partes en cada línea
                continue  # Ignorar líneas que no siguen el formato esperado
            numero_documento = partes[1][:-4]  # Obtener el número de documento eliminando la extensión .txt
            try:
                similitud = float(partes[-1])  # Convertir la similitud a un número decimal
            except ValueError:
                continue  # Ignorar líneas donde la similitud no es un número válido
            documentos_relevantes[numero_documento] = similitud
    return documentos_relevantes

# Leer documentos relevantes
documentos_relevantes = leer_documentos_relevantes(r'C:\Users\usuario\Fer-Pc\Escritorio\EPN\2024-A\SEPTIMO_SEMESTRE\RECUPERACION_DE_INFORMACION\repoMantillaRI\ProyectoRI\data\catslimpia.txt')
#documentos_relevantes = leer_documentos_relevantes('C:\Users\usuario\Fer-Pc\Escritorio\EPN\2024-A\SEPTIMO_SEMESTRE\RECUPERACION_DE_INFORMACION\repoMantillaRI\ProyectoRI\data\catslimpia.txt.txt')

def calcular_precision_recall(resultados_recuperados, resultados_relevantes):
    num_resultados_recuperados = len(resultados_recuperados)
    num_resultados_relevantes = len(resultados_relevantes)
    
    resultados_comunes = set(resultados_recuperados) & set(resultados_relevantes)
    num_resultados_comunes = len(resultados_comunes)
    
    precision = num_resultados_comunes / num_resultados_recuperados if num_resultados_recuperados > 0 else 0
    recall = num_resultados_comunes / num_resultados_relevantes if num_resultados_relevantes > 0 else 0
    
    return precision, recall

def leer_consultas(archivo):
    consultas = []
    with open(archivo, 'r') as file:
        for line in file:
            consulta = line.strip()
            consultas.append(consulta)
    return consultas

# Leer consultas
ruta_archivo_consultas = r'C:\Users\usuario\Fer-Pc\Escritorio\EPN\2024-A\SEPTIMO_SEMESTRE\RECUPERACION_DE_INFORMACION\repoMantillaRI\ProyectoRI\data\querys.txt'
consultas = leer_consultas(ruta_archivo_consultas)

# Realizar búsqueda para cada consulta
resultados_totales = {}  # Almacenar los resultados de todas las consultas
for consulta in consultas:
    resultados_recuperados = engine.search(consulta)
    resultados_totales[consulta] = resultados_recuperados

# Calcular precisión y recall para cada consulta
precisiones = []
recalls = []
for consulta, documentos_recuperados in resultados_totales.items():
    precision, recall = calcular_precision_recall(documentos_recuperados, documentos_relevantes)
    precisiones.append(precision)
    recalls.append(recall)

# Calcular la precisión y el recall promedio para todas las consultas
precision_promedio = sum(precisiones) / len(precisiones)
recall_promedio = sum(recalls) / len(recalls)

print("Precisión promedio:", precision_promedio)
print("Recall promedio:", recall_promedio)