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


