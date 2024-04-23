import openai
from open_ai_utils import simular_respuesta_generativa, printear_tabla_generativamente
from LLMsToAnomalyDetection import ADAgent
import numpy as np
from time import sleep

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

import os
import pickle
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
    agent = ADAgent(
        data_file= 'HVAC_test_processeded.csv', 
        data_path= '../data/Anomaly_detection/processed_data/',
        models_path= './models/anomaly_detection/')

    simular_respuesta_generativa('AGENTE:\nHOLA! Soy tu agente experto en detección de anomalías en un sistema de calefacción y ventilación!.\n\n')
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
                # printeamos la consulta del usuario
                simular_respuesta_generativa(f'\nPROMPT_USUARIO_{agent.historico.contador_interacciones+1}:\n\n{consulta_usuario}\n\n')

                if agent.historico.contador_interacciones <1: 

                    simular_respuesta_generativa(f'\nAGENTE:\nEl sistema contiene {len(agent.data)} lecturas de datos:\
                          \nDesde la fecha: {agent.data.index.min()}, hasta: {agent.data.index.max()}\n\n')
                    sleep(0.5)
                    simular_respuesta_generativa(f'\nAGENTE:\nMe dispongo a cargar los modelos para hacer inferencia...:\n\n')
                    sleep(0.5)
                    agent.cargar_modelos()
                    simular_respuesta_generativa(f'Los modelos se han cargado correctamente!\n\n')
                    sleep(1)

                # le pedimos la consulta al agente y printeamos la tabla
                simular_respuesta_generativa('Me dispongo a cargar los datos para atender tu consulta...\n\n')
                sleep(0.5)
                argumentos_para_cargar_datos = agent.map_prompt_to_load_data(consulta_usuario=consulta_usuario)
                agent.lectura_de_datos(**argumentos_para_cargar_datos)
                simular_respuesta_generativa('Tus datos se han cargado correctamente!\n\n')
                sleep(0.5)
                if len(agent.lectura_datos_base) >12: 

                    simular_respuesta_generativa(f'\nAGENTE:\nAqui tienes una muestra de la lectura de datos que has pedido...: \n\n')
                    printear_tabla_generativamente(agent.lectura_datos_base.head(12).to_markdown())
                    print('\n')
                    
                else: 
                    simular_respuesta_generativa(f'\nAGENTE:\nAqui tienes la lectura de datos que has pedido...: \n\n')
                    printear_tabla_generativamente(agent.lectura_datos_base.to_markdown())
                    print('\n')

                sleep(1)
                simular_respuesta_generativa(f'\nAGENTE:\nVoy comprobar el estado del sistema...:\n\n') 
                predictions = agent.predict_anomalies()
                simular_respuesta_generativa(f'Estado del sistema comprobado con éxito!:\n\n') 
                simular_respuesta_generativa(f'\nAGENTE:\nAqui tienes las predicciones para tu muestra de datos:\n\n') 

                if len(agent.lectura_datos_base) >12: 
                    simular_respuesta_generativa(predictions['anomaly_grade'].head(12).to_markdown())
                else: 
                    simular_respuesta_generativa(predictions['anomaly_grade'].to_markdown())

                # generamos el informe de la consulta dandole al agente la tabla el codigo sql y la consulta del usuario
                simular_respuesta_generativa('\nAGENTE:\nMe dispongo a generar el informe de tu consulta...\n\n')

                informe = agent.informe_resultado(
                        consulta_usuario=consulta_usuario, 
                        max_tokens_respuesta=1500)
                    
                # printeamos el informe
                simular_respuesta_generativa(f'\nAGENTE:\nAqui tienes el informe de tu consulta: \n{informe}\n\n')

            # continuamos chat sobre la lectura o nueva consulta
            simular_respuesta_generativa('\nAGENTE:\n¿Tienes alguna nueva consulta o duda sobre la consulta anterior?\n\n')
            consulta_usuario = str(input())
            conversacion, nueva_consulta = agent.continuar_conversacion(usuario= consulta_usuario)

            while conversacion == True and nueva_consulta== False:

                # printeamos la consulta del usuario sobre respuestas anteriores
                simular_respuesta_generativa(f'\nPROMPT_USUARIO_{agent.historico.contador_interacciones +1}:\n{consulta_usuario}\n\n')
                respuesta_agente = agent.pregunta_sobre_consulta_anterior(
                    usuario= consulta_usuario,
                    max_tokens_respuesta=1000
                    )
                
                # printeamos las nuevas respuesta del sistema
                simular_respuesta_generativa(f'\nAGENTE:\nLa respuesta a tu consulta:\n{respuesta_agente}\n\n')

                # continuamos chat sobre la lectura o nueva consulta
                simular_respuesta_generativa('\nAGENTE:\n¿Tienes alguna nueva consulta o duda sobre la consulta anterior?\n\n')
                consulta_usuario = str(input())
                conversacion, nueva_consulta = agent.continuar_conversacion(
                    usuario= consulta_usuario,
                    max_tokens_historico= 1500
                    )

        simular_respuesta_generativa('AGENTE:\nEstas seguro de que quieres cerrar el chat? (Y/N)\n\n')
        ultima_oportunidad = str(input())

        if ultima_oportunidad.lower() == 'y': 
            finalizar = True

    simular_respuesta_generativa(f'\n\nHa sido un placer ayudarte. Hasta la próxima!!')
    agent.historico_completo.almacenar_historico_txt(nombre_archivo= 'historico_AD_')

if __name__=='__main__':
    main()
