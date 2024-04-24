from LLMstoChat import ChatAgent
from LLMsToAnomalyDetection import ADAgent
from LLMstoDataBase import SQLAgent
from LLMsToOCR import OCRAgent
from time import sleep

from sentence_transformers import SentenceTransformer

import pandas as pd 
import numpy as np 


import warnings
warnings.filterwarnings('ignore')

from ultralytics import YOLO
import os 
import cv2
import openai 
from dotenv import load_dotenv
from open_ai_utils import enviar_promt_chat_completions_mode, simular_respuesta_generativa, cosine_similarity, printear_tabla_generativamente
from flujo_de_dialogo import HistoricoConversacion

# Cargar variables de entorno desde .env
load_dotenv()

# Acceder a la API key
api_key = os.getenv("API_KEY")
openai.api_key = api_key

def SQL(consulta_usuario:str): 

    finalizar = False
    primera_consulta = True

    rental_bbdd = {
        'bbdd_name':'dvdrental', 
        'user'     :'postgres', 
        'password' :'123', 
        'host'     :'localhost', 
        'port'     :'5432'
    }

    # inicializamos el agente
    agent = SQLAgent(**rental_bbdd)

    # entramos en la conversación-chat
    while finalizar == False:

        conversacion = True
        nueva_consulta = True

        ### podria hacer una funcion para reconocer a que base de datos se quiere conectar el usuario

        # el usuario hace la consulta
        simular_respuesta_generativa('AGENTE:\nSoy tu agente experto en bases de datos.\n\n')
        #consulta_usuario = str(input())

        while conversacion: 

            # quizas este if no hace falta
            if nueva_consulta== True:   
                # printeamos la consulta del usuario
                #simular_respuesta_generativa(f'\nPROMPT_USUARIO_{agent.historico.contador_interacciones+1}:\n\n{consulta_usuario}\n\n')

                # le pedimos la consulta al agente y printeamos la tabla
                tabla_resultado, codigo_sql, error = agent.nlp_to_sql(
                        consulta_nlp=consulta_usuario, 
                        metadata_token_limit= 1000, 
                                )
                intentos_resolver_error = 0    

                # si el codigo SQL da error se intentara resovler hasta 3 veces
                while error == 'SQL': 

                    tabla_resultado, codigo_sql, error = agent.resolver_error_SQL(
                        consulta_nlp=consulta_usuario, 
                        codigo_sql_conflictivo= codigo_sql, 
                        mensaje_error= tabla_resultado
                            )
                    if intentos_resolver_error >= 3: break
                        
                    intentos_resolver_error+= 1

                simular_respuesta_generativa(f'\nAGENTE:\nAqui tienes el resultado de tu consulta:\n\n') # \n{tabla_resultado}
                print(tabla_resultado)
                print('\n')

                # generamos el informe de la consulta dandole al agente la tabla el codigo sql y la consulta del usuario
                simular_respuesta_generativa('\nAGENTE:\nMe dispongo a generar el informe de tu consulta...\n\n')
                informe = agent.informe_resultado(
                        consulta_usuario= consulta_usuario, 
                        tabla_texto=tabla_resultado, 
                        max_tokens_respuesta=1000, 
                        codigo_sql= codigo_sql )
                nueva_consulta= False
                    
                # printeamos el informe
                simular_respuesta_generativa(f'\nAGENTE:\nAqui tienes el informe de tu consulta: \n{informe}\n\n')

            # continuamos chat sobre la lectura o nueva consulta
            simular_respuesta_generativa('\nAGENTE:\n¿Tienes alguna nueva consulta o duda sobre la consulta anterior?\n\n')
            consulta_usuario = str(input())
            conversacion, nueva_consulta = agent.continuar_conversando(
                usuario= consulta_usuario, 
                tabla_consulta_anterior=tabla_resultado,
                codigo_sql_ejecutado=codigo_sql
                )

            while conversacion == True and nueva_consulta== False:

                # printeamos la consulta del usuario sobre respuestas anteriores
                # simular_respuesta_generativa(f'\nPROMPT_USUARIO_{agent.historico.contador_interacciones +1}:\n{consulta_usuario}\n\n')

                respuesta_agente = agent.pregunta_sobre_consulta_anterior(
                    usuario= consulta_usuario, 
                    tabla_consulta_anterior=tabla_resultado, 
                    consulta_sql_anterior=codigo_sql
                    )
                
                # printeamos las nuevas respuesta del sistema
                simular_respuesta_generativa(f'\nAGENTE:\nLa respuesta a tu consulta:\n{respuesta_agente}\n\n')

                # continuamos chat sobre la lectura o nueva consulta
                simular_respuesta_generativa('\nAGENTE:\n¿Tienes alguna nueva consulta o duda sobre la consulta anterior?\n\n')
                consulta_usuario = str(input())
                conversacion, nueva_consulta = agent.continuar_conversando(
                    usuario= consulta_usuario, 
                    codigo_sql_ejecutado=codigo_sql, 
                    tabla_consulta_anterior=tabla_resultado, 
                    max_tokens_historico= 1500
                    )

        simular_respuesta_generativa('AGENTE:\nEstas seguro de que quieres cerrar el chat? (Y/N)\n\n')
        ultima_oportunidad = str(input())

        if ultima_oportunidad.lower() == 'y': 
            finalizar = True
            agent.close_connection()

    simular_respuesta_generativa(f'\n\nHa sido un placer ayudarte. Hasta la próxima!!')
    agent.historico_completo.almacenar_historico_txt(nombre_archivo= 'historico_SQL_')

def AD(consulta_usuario:str): 

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
        #consulta_usuario = str(input())

        while conversacion: 

            # quizas este if no hace falta
            if nueva_consulta== True:   
                # printeamos la consulta del usuario
                # simular_respuesta_generativa(f'\nPROMPT_USUARIO_{agent.historico.contador_interacciones+1}:\n\n{consulta_usuario}\n\n')

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
                #simular_respuesta_generativa(f'\nPROMPT_USUARIO_{agent.historico.contador_interacciones +1}:\n{consulta_usuario}\n\n')
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


def OCR(consulta_usuario:str): 

    finalizar = False
    # inicializamos el agente
    agent = OCRAgent()

    simular_respuesta_generativa('AGENTE:\nHOLA! Soy tu agente de OCR experto en extraer texto de documentos y detectar objetos manuscritos.\n\n')
    simular_respuesta_generativa('Me dispongo a ejecutar tu consulta.\n\n')
    # entramos en la conversación-chat
    while finalizar == False:

        conversacion = True
        nueva_consulta = True

        # el usuario hace la consulta
        # simular_respuesta_generativa('\n¿Dime, en qué quieres que te ayude?\n\n')
        # consulta_usuario = str(input())

        while conversacion: 

            # quizas este if no hace falta
            if nueva_consulta== True:   
                # printeamos la consulta del usuario
                #simular_respuesta_generativa(f'\nPROMPT_USUARIO_{agent.historico.contador_interacciones+1}:\n\n{consulta_usuario}\n\n')


                document = agent.choose_document()
                simular_respuesta_generativa(f'\nAGENTE:\n Se ha cargado el siguente documento: {document}\n')
                #agent.show_document()
                sleep(0.5)
                simular_respuesta_generativa(f'\nMe dispongo a extraer el texto del documento...:\n\n')
                sleep(0.5)
                agent.extract_text_from_docu()
                simular_respuesta_generativa(f'\nProcedo a detectar objetos manuscritos...:\n\n')
                agent.signature_detection()
                sleep(0.5)

                simular_respuesta_generativa('\nAGENTE:\nMe dispongo a generar el informe del documento...\n\n')

                informe = agent.informe_resultado(
                        consulta_usuario=consulta_usuario, 
                        max_tokens_respuesta=2000)
                    
                # printeamos el informe
                simular_respuesta_generativa(f'\nAGENTE:\nAqui tienes el informe de tu consulta: \n{informe}\n\n')

            # continuamos chat sobre la lectura o nueva consulta
            simular_respuesta_generativa('\nAGENTE:\n¿Tienes alguna nueva consulta o duda sobre la consulta anterior?\n\n')
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
    agent.historico_completo.almacenar_historico_txt(nombre_archivo= 'historico_OCR_')

def CHAT(consulta_usuario:str):

    finalizar = False

    # inicializamos el agente
    agent = ChatAgent()

    simular_respuesta_generativa('AGENTE:\nHOLA! Soy tu agente de Chat.\n\n')
    # entramos en la conversación-chat
    while finalizar == False:

        conversacion = True
        nueva_consulta = True

        # el usuario hace la consulta
        # simular_respuesta_generativa('\n¿Dime, en qué quieres que te ayude?\n\n')
        # consulta_usuario = str(input())

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

def main(): 
    model_saved_path = 'C:/Users/plane/OneDrive/Escritorio/COMPUTING SCIENCE/TFM_transformer_agents/src/models/SBERT'
    decision_model = SentenceTransformer(model_saved_path)
    simular_respuesta_generativa('Hola, en que puedo ayudarte\n\n')

    consulta_usuario = str(input())

    SQL_decr = 'Transforma consultas en lenguaje natural a código SQL, ejecuta consultas en la base de datos y proporciona informes detallados de los resultados en lenguaje natural.'
    HVAC_descr = 'Analiza datos de sistemas HVAC para detectar anomalías usando modelos de machine learning, preprocesa datos, realiza predicciones y entrega un informe detallado con observaciones relevantes y predicciones sobre el estado del sistema.'
    CHAT_descr =  'Interactúa en conversaciones generales, respondiendo a preguntas y proporcionando información o asistencia sobre temas variados que no requieren acceso a bases de datos específicas, análisis de anomalías o procesamiento de documentos.'
    OCR_descr= 'Procesa documentos para extracción de texto mediante OCR y detecta objetos escritos a mano como firmas, iniciales, redacciones o fechas usando un modelo YOLO ajustado, y ofrece una traducción y resumen del contenido destacando información clave e indicando la ubicación del contenido manuscrito.'


    user_embedding = decision_model.encode(consulta_usuario)
    SQL_embedding = decision_model.encode(SQL_decr)
    HVAC_embedding = decision_model.encode(HVAC_descr)
    OCR_embedding = decision_model.encode(OCR_descr)
    CHAT_embedding = decision_model.encode(CHAT_descr)

    sim_SQL= cosine_similarity(user_embedding, SQL_embedding)
    sim_HVAC= cosine_similarity(user_embedding, HVAC_embedding)
    sim_OCR= cosine_similarity(user_embedding, OCR_embedding)
    sim_chat= cosine_similarity(user_embedding, CHAT_embedding)

    sim_vec = pd.Series(data={'SQL':sim_SQL, 'AD':sim_HVAC, 'OCR':sim_OCR, 'CHAT':sim_chat})

    max_similarity_agent = sim_vec.index[sim_vec.argmax()]

    if max_similarity_agent == 'SQL': SQL(consulta_usuario=consulta_usuario)
    elif max_similarity_agent == 'AD': AD(consulta_usuario=consulta_usuario)
    elif max_similarity_agent == 'OCR': OCR(consulta_usuario= consulta_usuario)
    elif max_similarity_agent == 'CHAT': CHAT(consulta_usuario=consulta_usuario)

if __name__ == '__main__':
    main()