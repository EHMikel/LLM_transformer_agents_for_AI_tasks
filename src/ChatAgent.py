import openai
from open_ai_utils import simular_respuesta_generativa, printear_tabla_generativamente 
from LLMstoChat import ChatAgent

import numpy as np
from time import sleep

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

import os
from dotenv import load_dotenv
import warnings

    # Ignorar todos los warnings
warnings.filterwarnings('ignore')

    # Cargar variables de entorno desde .env
load_dotenv()

    # Acceder a la API key
api_key = os.getenv("API_KEY")
openai.api_key = api_key


def main():

    finalizar = False

    # inicializamos el agente
    agent = ChatAgent()

    simular_respuesta_generativa('AGENTE:\nHOLA! Soy tu agente de Chat.\n\n')
    # entramos en la conversación-chat
    while finalizar == False:

        conversacion = True
        nueva_consulta = True

        # el usuario hace la consulta
        simular_respuesta_generativa('\n¿Dime, en qué quieres que te ayude?\n\n')
        consulta_usuario = str(input())

        while conversacion: 

            # quizas este if no hace falta
            if nueva_consulta== True:   

                informe = agent.chatear(
                        consulta_usuario=consulta_usuario, 
                        max_tokens_respuesta=2000)
                    
                # printeamos el informe
                simular_respuesta_generativa(f'\nAGENTE:\n{informe}\n\n')

            # continuamos chat sobre la lectura o nueva consulta
            simular_respuesta_generativa('\nAGENTE:\n¿Necesitas hablar de algo más?\n\n')
            consulta_usuario = str(input())
            conversacion, nueva_consulta = agent.continuar_conversacion(usuario= consulta_usuario)

            while conversacion == True and nueva_consulta== False:

                # printeamos la consulta del usuario sobre respuestas anteriores
                #simular_respuesta_generativa(f'\nPROMPT_USUARIO_{agent.historico.contador_interacciones +1}:\n{consulta_usuario}\n\n')
                respuesta_agente = agent.pregunta_sobre_consulta_anterior(
                    usuario= consulta_usuario,
                    max_tokens_respuesta=1500
                    )
                
                # printeamos las nuevas respuesta del sistema
                simular_respuesta_generativa(f'\nAGENTE:\nLa respuesta a tu consulta:\n{respuesta_agente}\n\n')

                # continuamos chat sobre la lectura o nueva consulta
                simular_respuesta_generativa('\nAGENTE:\n¿Tienes alguna nueva consulta o duda sobre la consulta anterior?\n\n')
                consulta_usuario = str(input())
                conversacion, nueva_consulta = agent.continuar_conversacion(
                    usuario= consulta_usuario,
                    max_tokens_historico= 2000
                    )

        simular_respuesta_generativa('AGENTE:\nEstas seguro de que quieres cerrar el chat? (Y/N)\n\n')
        ultima_oportunidad = str(input())

        if ultima_oportunidad.lower() == 'y': 
            finalizar = True

    simular_respuesta_generativa(f'\n\nHa sido un placer ayudarte. Hasta la próxima!!')
    agent.historico_completo.almacenar_historico_txt(nombre_archivo= 'historico_CHAT_')

if __name__=='__main__':
    main()
