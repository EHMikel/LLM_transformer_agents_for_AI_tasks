from calcular_tokens import num_tokens_from_string
from open_ai_utils import enviar_promt_chat_completions_mode
import numpy as np
import os

class HistoricoConversacion:

    def __init__(self): 
        self.historico = ''                   # variable del historico en si
        self.contador_interacciones= 0        # el contador de las interacciones entre el agente y el usuario
        self.info_consultas_estructurada = {} # guardaremos los datos en un diccionario de listas

    def actualizar_historico(self, mensaje:str, role:str= 'agent', tipo:str= 'respuesta'): 
        '''Recibe el historico anterior y la nueva interaccion entre el usuario y el agente, y actualiza el historico. \
           Además, sube el valor del la variable contador_interacciones y lo devulve junto con el historico.'''
        
        if role.lower() == 'usuario' or role.lower() == 'user':
            self.contador_interacciones+= 1
            self.historico += f'prompt_{self.contador_interacciones} USUARIO:\n{mensaje}\n\n'
            
        else:
            self.historico += f'{tipo}_{self.contador_interacciones} AGENTE:\n{mensaje}\n\n'


    def guardar_consulta_estructurada(self, usuario:str, tabla:str= None, respuesta_llm:str= None, codigo_sql:str= None):
        '''Este metodo guarda todos los datos de la consulta en un diccionario de listas. 
           Luego me puede dar mucha flexibilidad para extraer información o incluso pasarlo a dataframe.
           Incluso podría calcular los embeddings
        '''
        consulta = [usuario, tabla, respuesta_llm, codigo_sql]
        # self.info_consultas_estructurada['consulta_'+str(self.contador_interacciones)] = consulta
        self.info_consultas_estructurada[self.contador_interacciones] = consulta


    def ventana_historico(self, max_tokens:int= 1000)->str:
        """
        Procesa el histórico de la conversación y acumula mensajes hasta alcanzar un límite de tokens, 
        para devolver una ventana de los ultimos max_tokens historico.
        """
        # Separar el histórico en palabras.
        palabras = self.historico.split(' ')[::-1]        # guardamos el historico de más reciente a más antiguo en una lista 
        historico_limitado = []                           # variable de ventana de historico
        tokens_acumulados = 0                             # variable de tokens acumulados

        for palabra in palabras:                                       # para cada palabra dentro del historico
            tokens_palabra = num_tokens_from_string(palabra)           # Calcular los tokens para la palabra actual.
            if tokens_acumulados + tokens_palabra > max_tokens: break  # si se excede el límite de tokens rompemos el bucle

            else:                                       # si no se excede el limite de tokens ...
                historico_limitado.append(palabra)      # Agregar la palabra al histórico limitado 
                tokens_acumulados += tokens_palabra     # aumenta la cantidad de tokens acumulados

        historico_limitado_str = ' '.join(historico_limitado[::-1]) # ahora unimos en str el historico en el orden original separado por un espacio
        self.ventana_ultimo_historico = historico_limitado_str      # guardamos esta ventana de historico como un atributo de instancia

        return historico_limitado_str                               # devolvemos la ventana de historico
    
    # guardamos el historico en un txt 
    def almacenar_historico_txt(self, nombre_archivo:str):
        ruta_chats = ruta_chats = 'chats/'
    
        # historico_completo = agent.historico_completo.historico
        nombre_historico = nombre_archivo + str(np.random.randint(low= 0, high= 1_000_000)) +'.txt'
        ruta_historico_chat = os.path.join(ruta_chats, nombre_historico)
        with open(ruta_historico_chat, 'w', encoding='utf-8') as archivo_historico:
            archivo_historico.write(self.historico)

# OTRAS IDEAS
    # guardamos la instancia del agente
    # ruta_agente = os.path.join(ruta_chats, 'agenteSQL.pickle')
    # with open(ruta_agente, 'wb') as guardar_agente:
    #     pickle.dump(agent, guardar_agente)

    # guardamos el historico en un txt 
    # historico_completo = agent.historico_completo.historico
    # nombre_historico = 'historico_SQL_' + str(np.random.randint(low= 0, high= 1_000_000)) +'.txt'
    # ruta_historico_chat = os.path.join(ruta_chats, nombre_historico)
    # with open(ruta_historico_chat, 'w', encoding='utf-8') as archivo_historico:
    #     archivo_historico.write(historico_completo)


def continuar_conversacion_AD(respuesta_usuario)-> bool:
    '''Clasifica el prompt del usuario de dos maneras: 
       1.- Si este quiere seguir conversando 'continuar' = True o False.
       2.- Si se trata de una nueva consulta o si es una consulta anterior 'nueva_consulta' = True o False'''

    tools = [
    {
    "type": "function",
    "function": {
        "name": "continuar_conversacion",
        "description": "Debes extraer DOS PARAMETROS clasificando una respuesta del usuario; por un lado:\n\
                        tienes que clasificar el parametro 'continuar':\n  \
                        - True (seguir chateando) si el usuario hace una pregunta o tiene alguna petición.\n \
                        - False(no seguir chateando) si el usuario no tiene ninguna pregunta o petición.\n \
                        por otro lado, si 'continuar'es True; debes clasificar el parametro 'nueva_cosulta':\n  \
                        - True (es una nueva consulta) el usuario pide una lectura sobre datos de otro momento o una nueva consulta.\n \
                        - False(es una cuestion sobre una lectura anterior) el usuario pide explicaciones sobre una consulta anterior.\n \
                        Ambos parametros son booleanos. Si 'continuar' es False, entonces 'nueva_consulta' también es False",
        "parameters": {
            "type": "object",
            "properties": {
                "continuar": {
                    "type": "boolean",
                    "description": "solo puede ser True o False"
                            },
                'nueva_consulta':{
                    "type": "boolean",
                    "description": "solo puede ser True o False"
                            },
            },
            "required": ["continuar", "nueva_consulta"]
            }
        }
    }
]

    extraccion_argumentos = [
        {'role': 'system', 'content': f'Tu objetivo es extraer los argumentos necesarios para ejecutar la función que te he pasado en tools'}, 
        {'role': 'user', 'content': f'{respuesta_usuario}'} ]

    argumentos_extraidos_del_llm = enviar_promt_chat_completions_mode(
            mensaje=extraccion_argumentos, 
            funciones= tools, 
            forzar_funciones= {"type": "function", "function": {"name": "continuar_conversacion"}}, 
            aleatoriedad= 0, 
            probabilidad_acumulada=1)
    
    return argumentos_extraidos_del_llm['continuar'], argumentos_extraidos_del_llm['nueva_consulta']


def pregunta_sobre_consulta_anterior_AD(consulta_usuario:str, historico:str, datos_base:str, 
                                        datos_descr:str, predicciones:str, max_tokens_respuesta:int= 1000): 
    
    '''Recibe una pregunta del usuario sobre una respuesta anterior del sistema y el agente devuelve una respuesta teniendo \
       acceso al historico y a la los datos de lectura, descripción y predicciones'''

    prompt_conversacion = [
    {'role': 'system', 'content': f'Eres un asistente de ayuda para un diagnostico de fallos en un sistema energético HVAC que responde de manera concisa sobre un \
                                    informe generado previamente. Siempre pones las unidades de las variables. El historico de la conversación: {historico}\
                                    recuerda: 1 significa valor típico y -1 valor atípico. Tu respuesta debe ser como maximo de {max_tokens_respuesta-100} tokens'}, 
                                
    {'role': 'user', 'content': f'A partir de la siguiente lectura:\n {datos_base}\n, con descripción: \n {datos_descr}\n y la siguiente \
                                  predicción: \n {predicciones}\n responde a mi consulta: \n{consulta_usuario}'}]
    
    respuesta = enviar_promt_chat_completions_mode(
            mensaje= prompt_conversacion, 
            modelo="gpt-4-1106-preview", 
            maximo_tokens=max_tokens_respuesta, 
            aleatoriedad=0.2, 
            probabilidad_acumulada=0.8)

    return respuesta