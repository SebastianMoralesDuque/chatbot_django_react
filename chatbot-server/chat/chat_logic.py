import string
import time
import numpy as np
import tensorflow as tf
import unicodedata
from tensorflow.keras.preprocessing.text import Tokenizer
import json
import Levenshtein
import os
file_path = os.path.join(os.path.dirname(__file__), 'intents.json')
file_path2 = os.path.join(os.path.dirname(__file__), 'modelo_chatbot_final.h5')


# Cargar datos del archivo JSON
with open(file_path, encoding='utf-8') as archivo:
    datos = json.load(archivo)

# Preprocesamiento de los datos
entrenamiento = []
clases = []
documentos = []
ignorar = ["?", "!", ".", ","]

for intent in datos["intents"]:
    for patron in intent["patterns"]:
        # Convertir a minúsculas y eliminar signos de puntuación
        palabras = [palabra.lower() for palabra in patron.split() if palabra not in ignorar]
        entrenamiento.append(" ".join(palabras))
        clases.append(intent["tag"])
        documentos.append((palabras, intent["tag"]))

# Crear diccionario de palabras
tokenizer = Tokenizer()
tokenizer.fit_on_texts(entrenamiento)
palabras = tokenizer.word_index
num_palabras = len(palabras) + 1

#cargar modelo
modelo = tf.keras.models.load_model(file_path2)



def procesar_entrada(entrada):
    # Eliminar signos de puntuación y tildes
    entrada = entrada.lower()
    entrada = entrada.translate(str.maketrans('', '', string.punctuation))
    entrada = unicodedata.normalize('NFKD', entrada).encode('ASCII', 'ignore').decode('utf-8')
    return entrada

last_message = ""

def logic(texto):
    global last_message
    
    # Procesa la entrada del usuario para quitar tildes y signos
    texto = procesar_entrada(texto)
    
    # Verifica que el usuario no haya escrito lo mismo antes
    if texto.lower() in last_message.lower():
        return "Ya has dicho eso antes. ¿Hay algo más en lo que pueda ayudarte?"
    
    last_message = texto
    # Preprocesamiento de la entrada
    entrada = [0] * num_palabras
    palabras_entrada = [palabra.lower() for palabra in texto.split() if palabra not in ignorar]
    for palabra in palabras_entrada:
        if palabra in palabras:
            entrada[palabras[palabra]] = 1

    # Predecir respuesta con modelo
    prediccion = modelo.predict(np.array([entrada]))
    respuesta_index = np.argmax(prediccion)
    tag_respuesta = clases[respuesta_index]

    # Asociar pregunta con respuesta
    preguntas_respuestas = {}
    for intent in datos["intents"]:
        if intent["tag"] == tag_respuesta:
            patterns = intent["patterns"]
            responses = intent["responses"]
            if len(patterns) == len(responses):
                for i, pattern in enumerate(patterns):
                    pregunta = procesar_entrada(pattern)
                    respuesta = responses[i]
                    preguntas_respuestas[pregunta] = respuesta

    # Seleccionar respuesta
    if preguntas_respuestas:
        respuesta = preguntas_respuestas.get(texto.lower(), "")
        if respuesta == "":
            preguntas = []
            respuestas = []
            for intent in datos["intents"]:
                if intent["tag"] == tag_respuesta:
                    patterns = intent["patterns"]
                    responses = intent["responses"]
                    if len(patterns) == len(responses):
                        preguntas.extend([procesar_entrada(pattern) for pattern in patterns])
                        respuestas.extend(responses)
            distancias = [Levenshtein.distance(texto.lower(), pregunta.lower()) for pregunta in preguntas]
            pregunta_similar = preguntas[np.argmin(distancias)]
            respuesta = respuestas[np.argmin(distancias)]
    else:
        respuesta = np.random.choice(responses)

    return respuesta
