import openai                                                               # importamos 
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from time import sleep

# Cargar variables de entorno desde .env
load_dotenv()


# Acceder a la API key
api_key = os.getenv("API_KEY")
openai.api_key = api_key


def enviar_promt_completions_mode(
        mi_prompt: str, model: str= "gpt-3.5-turbo-instruct", temp= 1, max_tokens: int= 500, 
        probabilidad_acumulada: float= 0.2, frequency_penalty= 0, presence_penalty= 0):
    
    # completions es la seccion de openAI para completar texto
    respuesta = openai.completions.create(
        model= model,                           # indicamos el modelo que vamos a utlizar
        prompt= mi_prompt,                      # el mensaje que le pasamos a chat 
        temperature= temp,                      # regula el grado de aleatoriedad de la respuesta 0 (siempre la misma respuesta), 1 (respuesta con aleatoriedad)
        max_tokens= max_tokens,                 # el maximo de token que queremos generar por cada promt 
        top_p= probabilidad_acumulada,          # delimitamos el universo de tokens de los cuales puede elegir para responder 1 = analiza todos 0.1 solo el 10% con mayor probabilidad, etc

        # se mueven en un rango de -2,2 
        frequency_penalty=frequency_penalty,    # si repiten tokens recibe penalización  
        presence_penalty= presence_penalty      # con que un token aparezca una vez ya recibe penalización
    )

    return respuesta.choices[0].text#.strip()    # el indice donde esta la respuesta de nuestro modelo


def enviar_promt_chat_completions_mode(
        mensaje: list, modelo: str = "gpt-4-1106-preview", formato: dict = None, 
        maximo_tokens: int= 500, aleatoriedad: float= 0.5, probabilidad_acumulada: float= 0.5):
     
    respuesta = openai.chat.completions.create(
        messages= mensaje, 
        model= modelo, 
        response_format= None, 
        max_tokens=maximo_tokens, 
        temperature=aleatoriedad, 
        top_p= probabilidad_acumulada
    )

    if formato == {'type': 'json_object'}: 
        return respuesta['choices'][0]['message']['content']
    
    else: return respuesta.choices[0].message.content


def get_embedding(texto, model= "text-embedding-ada-002") -> list:
    text = texto.replace('\n', ' ')
    respuesta = openai.embeddings.create(input= text, model= model)
    return respuesta.data[0].embedding


def cosine_similarity(embedding1, embedding2) -> float: 
    #print(type(embedding1), type(embedding2))
    from numpy.linalg import norm
    cos_sim = np.dot(embedding1, embedding2)/(norm(embedding1)*norm(embedding2))
    return cos_sim


def simular_respuesta_generativa(respuesta:str): 
    pos_ini = 0
    pos_fin = 0
    while pos_fin < len(respuesta): 
        pos_fin = pos_ini + np.random.randint(low= 1, high= 20)
        print(respuesta[pos_ini:pos_fin], end= '')
        pos_ini= pos_fin
        sleep(np.random.uniform(low= 0, high= 0.2))