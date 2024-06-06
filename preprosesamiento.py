import os
import re
import nltk
import pandas as pd
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup

# Configuraciones iniciales
stemmer = SnowballStemmer('english')

# Leer stop words personalizadas desde el archivo
def load_stopwords(file_path):
    with open(file_path, 'r') as file:
        stopwords = set(file.read().splitlines())
    return stopwords

# Ruta a los archivos de Reuters-21578 y stopwords
data_path = r'D:\U\7. Septimo\Rec. Info\test_txt'  # Cambia esto a la ruta correcta
stopwords_path = r"D:\U\7. Septimo\Rec. Info\reuters\reuters\stopwords.txt"  # Cambia esto a la ruta correcta


# Verificar que la ruta del archivo de stop words es correcta
print(f'Loading data from: {data_path}')
print(f'Loading stop words from: {stopwords_path}')

if not os.path.isfile(stopwords_path):
    print(f"Error: El archivo {stopwords_path} no existe.")
else:
    # Cargar las stop words personalizadas
    stop_words = load_stopwords(stopwords_path)

    # Función para limpiar y preprocesar el texto
    def preprocess_text(text):
        # Convertir a minúsculas
        text = text.lower()
        # Eliminar caracteres no deseados
        text = re.sub(r'\W+', ' ', text)
        # Tokenización
        tokens = word_tokenize(text)
        # Eliminar stop words y aplicar stemming
        processed_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
        return ' '.join(processed_tokens)

    # Función para extraer el contenido relevante
    def extract_content(file_content):
        soup = BeautifulSoup(file_content, 'html.parser')
        texts = soup.find_all('body')  # Esto es un ejemplo, ajusta según el formato real
        return ' '.join(text.get_text() for text in texts)

    # Ejemplo de lectura y procesamiento de archivos
    documents = []

    for file_name in os.listdir(data_path):
        if file_name.endswith('.sgm'):
            with open(os.path.join(data_path, file_name), 'r', encoding='latin1') as file:
                content = file.read()
                # Extraer el contenido relevante
                extracted_text = extract_content(content)
                cleaned_text = preprocess_text(extracted_text)
                documents.append(cleaned_text)

    # Guardar el resultado en un DataFrame de pandas
    df = pd.DataFrame(documents, columns=['text'])
    df.to_csv('preprocessed_reuters.csv', index=False)
