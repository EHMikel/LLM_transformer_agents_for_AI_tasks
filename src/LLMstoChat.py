
import pandas as pd 
import numpy as np 


import warnings
warnings.filterwarnings('ignore')

from ultralytics import YOLO
import os 
import cv2
import openai 
from dotenv import load_dotenv
from open_ai_utils import enviar_promt_chat_completions_mode
from flujo_de_dialogo import HistoricoConversacion

# Cargar variables de entorno desde .env
load_dotenv()

# Acceder a la API key
api_key = os.getenv("API_KEY")
openai.api_key = api_key



class ChatAgent(): 
    def __init__(self, llm_model= "gpt-4-1106-preview", max_tokens_respuesta= 2000): 

        self.model = llm_model
        self.max_tokens_respuesta = max_tokens_respuesta
        self.historico_completo = HistoricoConversacion()  # incluye las tablas 
        self.historico = HistoricoConversacion()           # solo incluye las respuestas en lenguaje natural y las consultas


    def chatear(self, consulta_usuario:str, max_tokens_respuesta=2000)->str:

        mi_prompt = [
            {'role': 'system', 
                    'content': f'Eres un asistente Que tiene conversaciones sobre temas generales. Responde al usuario y le recurdas de manera cómica\
                                que cada prompt que envia al sistema le cuesta dinero. Eres libre de trolear al usuario'
                                }, 
            {'role': 'user',   'content': f'{consulta_usuario}'},
            ]

        respuesta = enviar_promt_chat_completions_mode(
            mensaje=mi_prompt,
            maximo_tokens=max_tokens_respuesta, 
            probabilidad_acumulada=0.5, 
            aleatoriedad=0.9)
        
        self.historico.actualizar_historico(mensaje=respuesta, role= 'agent', tipo= 'respuesta')
        self.historico_completo.actualizar_historico(mensaje=respuesta, role= 'agent', tipo= 'respuesta')

        return respuesta
    
    def continuar_conversacion(self, usuario:str, max_tokens_historico:int= 2000)-> bool:
        '''Clasifica el prompt del usuario de dos maneras: 
        1.- Si este quiere seguir conversando 'continuar' = True o False.
        2.- Si se trata de una nueva consulta o si es una consulta anterior 'nueva_consulta' = True o False
        '''

        ultimo_historico = self.historico.ventana_historico(max_tokens= max_tokens_historico)

        extraccion_argumentos = [
                {'role': 'system', 
                        'content': 
                        f'Tu objetivo es extraer los argumentos ("continuar" y "nueva_consulta") para\
                        ejecutar la función que te he pasado en tools. Primero debes diferenciar si el usuario quiere\
                        seguir conversando o no: True o False. Segundo debes diferenciar si el usuario tiene una nueva \
                        consulta o no, sobre la consulta anterior. Para ello aqui tienes el historico:\n{ultimo_historico}\n\
                        Si puedes responder la pregunta del usuario con la información que tienes "nueva_consulta= False" de lo contrario "nueva_consulta"= True'
                        },

                {'role': 'user', 
                    'content': f'{usuario}'}
            ]

        argumentos_extraidos_del_llm = enviar_promt_chat_completions_mode(
                mensaje=extraccion_argumentos, 
                funciones= ChatAgent.tools, 
                forzar_funciones= {"type": "function", "function": {"name": "continuar_conversacion"}}, 
                aleatoriedad= 0, 
                probabilidad_acumulada=1)
        
        return argumentos_extraidos_del_llm['continuar'], argumentos_extraidos_del_llm['nueva_consulta']
    

    def pregunta_sobre_consulta_anterior(self, usuario:str, max_tokens_respuesta:int= 2000, max_tokens_historico:int= 3000): 
        
        ultimo_historico = self.historico.ventana_historico(max_tokens= max_tokens_historico)

        self.historico.actualizar_historico(mensaje= usuario, role= 'user')
        self.historico_completo.actualizar_historico(mensaje= usuario, role= 'user')

        prompt_conversacion = [
        {'role': 'system', 'content': 
                        f'Eres un asistente tiene conversaciones sobre temas generales. Responde al usuario y le recurdas de manera cómica\
                        que cada prompt que envia al sistema le cuesta dinero. Eres libre de trolear al usuario. \
                        El historico de la conversación: {ultimo_historico}\
                        Tu respuesta debe ser como maximo de {max_tokens_respuesta-100} tokens'}, 
                                    
        {'role': 'user', 
                    'content': f'{usuario}'}
                                    
                                    ]
        
        respuesta = enviar_promt_chat_completions_mode(
                mensaje= prompt_conversacion, 
                modelo="gpt-4-1106-preview", 
                maximo_tokens=max_tokens_respuesta, 
                aleatoriedad=0.2, 
                probabilidad_acumulada=0.8)
        
        self.historico.actualizar_historico(mensaje=respuesta, role= 'agent', tipo= 'respuesta_consulta_anterior')
        self.historico_completo.actualizar_historico(mensaje=respuesta, role= 'agent', tipo= 'respuesta_consulta_anterior')

        return respuesta