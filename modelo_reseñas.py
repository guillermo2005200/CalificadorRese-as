import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Cargar el dataset
columnas = ["label", "title", "content"]
df = pd.read_csv("amazon_reviews.csv", names=columnas, header=None)

# Explorar los datos
print("Primeras filas del dataset:")
print(df.head())

print("\nCantidad de valores nulos por columna:")
print(df.isnull().sum())

# Eliminar filas con valores nulos en 'content'
df.dropna(subset=['content'], inplace=True)

# Descargar los recursos de NLTK si no están disponibles
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Cargar stopwords en inglés
stop_words = set(stopwords.words('english'))

# Función de preprocesamiento de texto
def limpiar_texto(texto):
    if isinstance(texto, str):
        texto = texto.lower()  # Convertir a minúsculas
        texto = re.sub(r'\W', ' ', texto)  # Eliminar caracteres especiales
        texto = re.sub(r'\s+', ' ', texto).strip()  # Eliminar espacios extra
        palabras = texto.split()  # Separar palabras sin usar word_tokenize()
        palabras = [palabra for palabra in palabras if palabra not in stop_words]  # Eliminar stopwords
        return " ".join(palabras)
    return ""

# Aplicar limpieza de texto
print("\nIniciando limpieza de texto...")
df["content_clean"] = df["content"].apply(limpiar_texto)
print("Limpieza de texto completada.")

# Verificar si hay valores vacíos después de la limpieza
print("\nValores nulos después de limpieza:")
print(df["content_clean"].isnull().sum())

# Representación numérica del texto con TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["content_clean"])
y = df["label"]

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Entrenar el modelo Naive Bayes
modelo_nb = MultinomialNB()
modelo_nb.fit(X_train, y_train)

# Evaluar el modelo
y_pred = modelo_nb.predict(X_test)
print("\nResultados del modelo:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Predicción con nuevas reseñas
nuevas_reseñas = [
    "This book describe the way to get the very best practices in software development. I recommend it for every one who want to develop better programs.",
    "This is a kind of programming bible. Every programmer must read this book to get good practices and to refresh the idea of how beauty it is to write beauty, clean and perfect code.",
    "As mentioned in other reviews, the book itself is low quality. The cover image is blurry, there is no print on the binding, and there are pointlessly large margins surrounding all of the text. On top of that, the printed font is grainy and difficult to read. I would rather track down a proper printing of this product, than try to trudge through reading this. As a comparison, Code Complete is three times longer than this book. However, I was able to easily hold and read that book. This version of Clean Code feels like you are holding a ream of printer paper, with every page printed in portrait mode. It is unwieldy to say the least."
]

nuevas_reseñas_clean = [limpiar_texto(texto) for texto in nuevas_reseñas]
X_nuevas = vectorizer.transform(nuevas_reseñas_clean)

predicciones = modelo_nb.predict(X_nuevas)

print("\nPredicciones para nuevas reseñas:")
for review, label in zip(nuevas_reseñas, predicciones):
    # Tomar las primeras 15 palabras de la reseña
    resumen_resena = " ".join(review.split()[:15]) + "..."
    print(f"Reseña: {resumen_resena} --> Predicción: {'Positiva' if label == 2 else 'Negativa'}")

print("\nFin del script.")

