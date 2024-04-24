import openai
from open_ai_utils import simular_respuesta_generativa, printear_tabla_generativamente
from LLMstoDataBase import SQLAgent
import numpy as np

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
        simular_respuesta_generativa('AGENTE:\nSoy tu agente experto en bases de datos. ¿En que te puedo ayudar hoy?\n\n')
        consulta_usuario = str(input())

        while conversacion: 

            # quizas este if no hace falta
            if nueva_consulta== True:   
                # printeamos la consulta del usuario
                simular_respuesta_generativa(f'\nPROMPT_USUARIO_{agent.historico.contador_interacciones+1}:\n\n{consulta_usuario}\n\n')

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
                simular_respuesta_generativa(f'\nPROMPT_USUARIO_{agent.historico.contador_interacciones +1}:\n{consulta_usuario}\n\n')

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

if __name__=='__main__':
    main()
