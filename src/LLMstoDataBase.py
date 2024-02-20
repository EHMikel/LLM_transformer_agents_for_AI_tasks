import openai                                                               # importamos 
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine, MetaData
import psycopg2
from open_ai_utils import cosine_similarity, get_embedding
import json
from calcular_tokens import num_tokens_from_string
import re
from open_ai_utils import enviar_promt_chat_completions_mode, simular_respuesta_generativa
from flujo_de_dialogo import HistoricoConversacion


# Cargar variables de entorno desde .env
load_dotenv()

# Acceder a la API key
api_key = os.getenv("API_KEY")
openai.api_key = api_key


class LLMsToDataBase:
    '''Clase en la cual se establecen las funcionalidades básicas para establecer una conexion entre 
       los LLMs y bases de datos SQL mediante postgreSQL. Estas funcionalidades serán extendidas a 
       otra clase que heredará de esta; SQLAgent
    '''

    # constructor de la clase
    def __init__(self, bbdd_name, user, password, host='localhost', port='5432'):
        self.engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{bbdd_name}")
        self.metadata = MetaData()
        self.bbdd_name = bbdd_name
        self.data_path = '../data/nlp_to_sql/'     # cambiar esta ruta por la ruta donde se guardaran los metadatos


    # cerrar la conexion con la base de datos
    def close_connection(self):
        self.engine.dispose()


    # pasar metadatos simples a string
    def simple_metadata_to_str(self) -> str:
        self.metadata.reflect(self.engine)
        tables_columns = {table.name: [column.name for column in table.columns] for table in self.metadata.tables.values()}
        return json.dumps(tables_columns, indent=4)
    

    # pasar de columna dataframe a string
    @staticmethod
    def get_string_from_metadata_df(metadata_df) -> str:
        # mi_string = '\n'.join(metadata_df['metadata_str'].fillna(''))
        return metadata_df['metadata_str'].fillna('').str.cat(sep= '\n')


    # pasar metadatos de dataframe a string
    def full_metadata_to_str(self) -> str:
        self.metadata.reflect(self.engine)
        metadata = {}

        for table in self.metadata.tables.values():
        # Por cada tabla, se almacena información detallada de sus columnas en el diccionario 'tablas_columnas'.
        # Se crea una clave en el diccionario para cada nombre de tabla.
            metadata[table.name] = {                                     
                "columnas": {column.name: {                           # dentro de cada tabla se crea un diccionario de 
                        "tipo": str(column.type),                     # Tipo de dato de la columna.
                        "nulo": column.nullable,                      # Booleano que indica si la columna acepta valores nulos.
                        "clave_primaria": column.primary_key,         # Booleano que indica si la columna es una clave primaria
                        "clave_foranea": bool(column.foreign_keys)}   # Booleano que indica si la columna es una clave foránea.
                    for column in table.columns}                      # Este bucle interno itera a través de todas las columnas de la tabla.
                    }

        return json.dumps(metadata, indent=4)


    # extraer metadatos básicos
    def simple_md_extraction_pipeline(self):
        '''Función para extraer los metadatos básicos de la base de datos; los nombre de cada tabla y 
           los nombres de cada columna de cada tabla.
        '''
        self.metadata.reflect(self.engine)
        metadatos = {}

        for table in self.metadata.tables.values():
            md = {}                                                               # dict de la tabla actual -> nombre_tabla:nombre_columnas
            col_names = [column.name for column in table.columns]                 # guardo los nombres de las columnas de cada tabla
            md[table.name] = col_names                                            # lo meto en el dict de la tabla actual
            md_str = json.dumps(md)                                               # lo paso a str
            metadata_str_tokens = num_tokens_from_string(md_str)                  # calculo los tokens de este str
            md_embedding = np.array(get_embedding(texto= md_str))                 # extraigo el embedding en formato np array
            metadatos[table.name] = [md_str, metadata_str_tokens, md_embedding]   # guardo el str, el embedding y los tokens 

        md_df = pd.DataFrame(metadatos)                                  # pasamos el diccionario a df
        md_df.index = ['metadata_str', 'md_str_tokens','embedding']      # manipulamos los indices
        md_df = md_df.T                                                  # cambiamos indices por columnas
        
        return md_df


    # extraer todos los metadatos de una base de datos
    def full_md_extraction_pipeline(self):
        '''Función para extraer todos los metadatos de la base de datos: para cada tabla se extre el nombre de la
           tabla, el nombre de cada columna, si esa columna es llave foranea o primaria, si acepta nulos o no, y 
           el tipo de dato que almacena cada columna. Es más costosa a nivel de token que la extracción simple
        '''

        self.metadata.reflect(self.engine)
        metadatos = {}

        for table in self.metadata.tables.values():
            metadatos_tabla = []  # Lista para almacenar los metadatos de cada tabla

            for column in table.columns:
                columna_metadatos = {
                    "col_name": column.name,                       # se guarda el nombre de la columna
                    "type": str(column.type),                      # Tipo de dato de la columna.
                    "null": column.nullable,                       # Booleano que indica si la columna acepta valores nulos.
                    "primary_key": column.primary_key,             # Booleano que indica si la columna es una clave primaria
                    "foreing_key": bool(column.foreign_keys)       # Booleano que indica si la columna es una clave foránea.
                }

                str_md_columna = json.dumps(columna_metadatos)  # se pasan los metadatos a str
                metadatos_tabla.append(str_md_columna)          # Añadir los metadatos y embedding de cada columna a la lista

            metadatos_tabla_str = 'table_name:' + table.name + '\n' + '\n'.join([str_md_col for str_md_col in metadatos_tabla])
            metadata_str_tokens = num_tokens_from_string(metadatos_tabla_str)
            
            md_embedding = np.array(get_embedding(texto=metadatos_tabla_str))                       # calculo el embedding de los metadatos
            metadatos[table.name] = [metadatos_tabla_str, metadata_str_tokens, md_embedding]        # Añadir los metadatos de la tabla al diccionario

        md_df = pd.DataFrame(metadatos)
        md_df.index = ['metadata_str', 'md_str_tokens','embedding']     # set_index(keys= ['metadata_str', 'embedding'])
        md_df = md_df.T                                                 #.reset_index(drop= True, inplace= True)
    
        return md_df

    # guardamos los metadatos en formato pickle
    def store_metadata_df_to_pickle(self):
        '''Esta función, dada una base de datos almacena los metadatos en formato pickle en
           diferentes niveles de profundidad
        '''
        names = [self.bbdd_name + '_simple_metadata.pickle', self.bbdd_name + '_full_metadata.pickle']

        for name in names:
            full_path = os.path.join(self.data_path, name)
            if not os.path.exists(full_path):

                # extraemos los metadatos simples
                if 'simple' in name:
                    simple_md_df = self.simple_md_extraction_pipeline()
                    simple_md_df.to_pickle(full_path)

                # extraemos los metadatos completos
                elif 'full' in name:
                    full_md_df = self.full_md_extraction_pipeline()
                    full_md_df.to_pickle(full_path)
                simular_respuesta_generativa(f"Archivo guardado: {full_path}\n")

            else:
                simular_respuesta_generativa(f'El archivo ya existe en: {full_path}\n')

        simular_respuesta_generativa(f"\nProceso de almacenamiento de metadatos para {self.bbdd_name} completado.\n\n")


    @staticmethod
    def search_most_similar_metadata(prompt:str, metadata, n_resultados:int= None, lim_tokens:int= None) -> pd.DataFrame: 
        '''Dado un prompt del usuario primero se calcula el embedding del prompt y se mide la similitud del coseno 
           en conparación a los metadatos correspondientes de cada columna de la base de datos, despues de ordenan 
           de mayor similitud a menor, y finalmente se devuelven todos los metadatos que entren en el limite de 
           tokens establecido y se devuelve un dataframe con los metadatos correspondientes.
        '''
        # se extrae el embedding de la pregunta del usuario
        prompt_embedding = get_embedding(prompt)

        # se saca la similitud de la pregunta con las posibles respuestas
        metadata['similarity']= metadata['embedding'].apply(lambda x: cosine_similarity(x, prompt_embedding))   

        # las ordenamos de mas similar a menos
        metadata = metadata.sort_values('similarity', ascending= False)   

        if n_resultados == None:           n_resultados = len(metadata)
        elif n_resultados > len(metadata): n_resultados = len(metadata)
        
        metadata_mas_similar = metadata.iloc[:n_resultados][['metadata_str', 'md_str_tokens','similarity', 'embedding']]
        metadata_mas_similar['token_cumsum'] = np.cumsum(metadata_mas_similar['md_str_tokens'])

        if lim_tokens == None: return metadata_mas_similar
        else:                  return metadata_mas_similar[metadata_mas_similar['token_cumsum'] <= lim_tokens]

    # recibe una consulta y mediante una plantilla especiífca gpt4 devuelve el código sql correspondiente
    @staticmethod
    def obtener_sql_de_la_consulta(consulta_nlp:str, metadatos_str:str) -> str:

        mi_prompt = [
            {'role': 'system', 
                      'content': f'Eres un asistente experto en bases de datos, que convierte peticiones de lenguaje natural a código SQL.\
                                   AQUI TIENES LOS METADATOS DE LA BASE DE DATOS: \n{metadatos_str}\n. Para expresiones regulares utiliza \
                                   "~" en lugar de "REGEXP".'},      
            {'role': 'user',   'content': f'{consulta_nlp}'},
            {'role': 'assistant', 'content': 'DEVUELVES SOLO EL CODIGO SQL SIN NINGUN COMENTARIO EN ESTE FORMATO: ```sql\nEL CODIGO SQL;\n```'}
            ]

        respuesta_sql = enviar_promt_chat_completions_mode(
            mensaje=mi_prompt, 
            probabilidad_acumulada=1, 
            aleatoriedad=0)
        
        return respuesta_sql
        
    # generar un informe de los resultados de nuestra consulta
    @staticmethod
    def informe_resultado(consulta_usuario:str, codigo_sql:str, tabla_texto: str, max_tokens_respuesta:int= 1000)->str:

        mi_prompt = [
            {'role': 'system', 
                     'content': f'Eres un asistente experto en generar informes detallados y concisos, sobre datos tabulares. \
                                El usuario previamente ha hecho una consulta en una base de datos y quiere un informe sobre \
                                el resultado de esa consulta, tu deber es generar un informe con la info más relevante. \
                                Además la tabla, viene dada por el siguiente codigo sql \n{codigo_sql}\n. \
                                La tabla que tienes que sintetizar es la siguiente:\n{tabla_texto}\n \
                                Debes acotar tu respuesta a un maximo de {max_tokens_respuesta-50} o menos'
                                }, 
            {'role': 'user',   'content': f'{consulta_usuario}'},
            ]

        respuesta_sql = enviar_promt_chat_completions_mode(
            mensaje=mi_prompt, 
            probabilidad_acumulada=0.8, 
            aleatoriedad=0.2)
        
        return respuesta_sql
     

class SQLAgent(LLMsToDataBase):
    '''Esta clase presenta las funcionalidades necesarias de la conversación con el LLM cuando le preguntemos sobre BBDD.
       Esta clase hereda de la anterior y tiene funcionalidades extra como hacer la consulta y generar un informe continuar conversando, 
       resolver dudas sobre una consulta anterior, intenta resolver los errores. Gestiona el histórico de la conversación internamente.
    '''

    # establecemos tools como variables de clase
    tools = [
    {
    "type": "function",
    "function": {
        "name": "continuar_conversacion",
        "description": 'Extrae los argumentos booleanos "continuar" y "nueva_consulta".\n\
         Si el usuario pregunta algo que no tiene nada que ver con posibles consultas a BBDDs, entonces continuar = "False"  \
         Si el usuario hace una pregunta que no puedes responder con el historico o la información de la tabla, entonces \
         "continuar"= "True", y "nueva_consulta" = "True". Si puedes responder a el prompt del usuario con la información del \
         historico o la tabla entonces "continuar"= "True", y "nueva_consulta" = "False"\
         Si "continuar" =  "False" entonces "nueva_consulta" = "False"\
         Debes diferenciar si el usuario quiere seguir converesando sobre la consulta anterior o tiene una nueva consulta',
        "parameters": {
            "type": "object",
            "properties": {
                "continuar": {
                    "type": "boolean",
                    "description": "solo puede ser True o False. Clasifica si el usuario quiere seguir conversando (sobre el tema) o no."
                            },
                'nueva_consulta':{
                    "type": "boolean",
                    "description": "solo puede ser True o False. Clasifica si el usuario tiene una nueva consulta (True) o no (False)"
                            },
            },
            "required": ["continuar", "nueva_consulta"]
                    }
                }
            }
        ]

    def __init__(self, bbdd_name=None, user=None, password=None, host='localhost', port='5432', llm_to_database=None):

        # Si no se proporciona una instancia, llama al constructor de la clase base con los argumentos proporcionados
        if llm_to_database is None:
            super().__init__(bbdd_name, user, password, host, port)

        # Si se proporciona una instancia, inicializa los atributos manualmente
        else:
            self.engine = llm_to_database.engine
            self.metadata = llm_to_database.metadata
            self.bbdd_name = llm_to_database.bbdd_name
            self.data_path = llm_to_database.data_path
        
        # cada agente tendrá su historico guardado y se almacenará como atributo de instancia
        self.historico_completo = HistoricoConversacion()  # incluye las tablas 
        self.historico = HistoricoConversacion()           # solo incluye las respuestas en lenguaje natural y las consultas

    # es un metodo interno que extrae el codigo sql y lo devuelve junnto con la tabla
    def _extract_sql_and_data(self,respuesta_sql:str):

        # buscamos codigo sql en la respuesta del llm
        regex_pattern = r"```sql\n(.*?;)\n```" 
        coincidencia = re.search(regex_pattern, respuesta_sql, re.DOTALL)


                # si regex encuentra codigo sql lo extrae
        if coincidencia:
            codigo_sql = coincidencia.group(1).strip()
            #print('codigo_sql')
            self.historico_completo.actualizar_historico(mensaje=codigo_sql, role= 'agent', tipo= 'codigo SQL')
            simular_respuesta_generativa(f'\nAGENTE:\nEl codigo sql que se ejecutará para responder tu consulta:\n\n{codigo_sql}\n\n')
                    
                # intentamos ejecutar la consulta con pandas y pasamos la tabla a texto para darsela despues al llm
            try: 
                df = pd.read_sql(codigo_sql, self.engine)
                df_text = df.to_markdown()
                self.historico_completo.actualizar_historico(mensaje=df_text, role= 'agent', tipo= 'tabla')
                error = None
                return df_text, codigo_sql, error
                    
                # si se me olvida levantar la base de datos 
            except psycopg2.OperationalError as e:
                mensaje_error= f'\nAGENTE:\nCon el 99.999999% de probabilidad se te ha olvidado levantar la base de datos, ESPABILA!!\n{e}'
                simular_respuesta_generativa(mensaje_error)
                error = 'BBDD'
                return None, None, error
                    
                    # si el codigo ejecutado en try da error
            except Exception as e:
                mensaje_error = f"\nAGENTE:\nError al ejecutar el SQL:\n\n{e}\n\n"
                simular_respuesta_generativa(f'\nAGENTE\n:{mensaje_error}')
                self.historico_completo.actualizar_historico(mensaje=mensaje_error, role= 'agent', tipo= 'error')
                error = 'SQL'
                return mensaje_error, codigo_sql, error
                    
                # si regex no encuentra codigo, devuelve el mensaje del llm
        else: 
            simular_respuesta_generativa(f'\nAGENTE:\nNo se encontro código SQL en la respuesta:\n\n{respuesta_sql}\n\n')
            self.historico_completo.actualizar_historico(mensaje=respuesta_sql, role= 'agent', tipo= 'error')
            error = 'REGEX'
            return None, respuesta_sql, error


    def nlp_to_sql(self, consulta_nlp:str, metadata_mode:str='simple', 
                    n_tablas:int=None, metadata_token_limit:int=2000):
        '''
        Esta función recibe una consulta en lenguaje natural y la formatea a codigo SQL para 
        luego devolver una tabla en formato texto y que la api de chat completions de openai 
        pueda procesar esa tabla. PROPORCIONA INFO DE LOS METADATOS DE LA BBDD AL AGENTE
        '''

        try: 
            # añadimos el primer mensaje 
            self.historico.actualizar_historico(mensaje=consulta_nlp, role= 'user')
            self.historico_completo.actualizar_historico(mensaje=consulta_nlp, role= 'user')

            # solo en la primera consulta se va ha asegurar de que los metadatos esten almacenados correctamente
            if self.historico.contador_interacciones <= 1:
                simular_respuesta_generativa(f'\nVoy a asegurar que los metadatos de bbdd {self.bbdd_name} esten disponibles...\n\n')
                self.store_metadata_df_to_pickle()

            if metadata_mode.lower() == 'full':
                metadata = pd.read_pickle(os.path.join(self.data_path, self.bbdd_name + '_full_metadata.pickle'))
            else:
                metadata = pd.read_pickle(os.path.join(self.data_path, self.bbdd_name + '_simple_metadata.pickle'))

            # se compara el embedding del prompt con los embeddings de los metadatos de cada tabla de la base de datos
            md_mas_similar = SQLAgent.search_most_similar_metadata(
                consulta_nlp, 
                metadata, 
                n_resultados= n_tablas, 
                lim_tokens= metadata_token_limit)

            # se pasan los metadatos a string
            metadatos_str = SQLAgent.get_string_from_metadata_df(md_mas_similar)
            self.metadata_str = metadatos_str

            # se envia la consulta nlp al llm para que devuelva el código sql
            respuesta_sql = SQLAgent.obtener_sql_de_la_consulta(consulta_nlp= consulta_nlp, metadatos_str= metadatos_str)

            # extrae el codigo sql de la respuesta y se pasa a tabla, tambien se gestionan los errores
            mensaje_o_tabla, codigo_sql, error = self._extract_sql_and_data(respuesta_sql=respuesta_sql)
            
            return mensaje_o_tabla, codigo_sql, error

                
        except Exception as e: 
            mensaje_error = f"\nAGENTE:\nLa consulta dio el siguiente error:\n\n{e}\n\n"
            simular_respuesta_generativa(mensaje_error)
            error = 'OTHER'
            return None, mensaje_error, error

    # Generamos informe de nuestra consulta
    def informe_resultado(self, consulta_usuario:str, codigo_sql:str, tabla_texto: str, max_tokens_respuesta:int= 1000)->str:

        mi_prompt = [
            {'role': 'system', 
                     'content': f'Eres un asistente experto en generar informes detallados y concisos, sobre datos tabulares. \
                                El usuario previamente ha hecho una consulta en una base de datos y quiere un informe sobre \
                                el resultado de esa consulta, tu deber es generar un informe con la info más relevante. \
                                Además la tabla, viene dada por el siguiente codigo sql \n{codigo_sql}\n. \
                                La tabla que tienes que sintetizar es la siguiente:\n{tabla_texto}\n \
                                Debes acotar tu respuesta a un maximo de {max_tokens_respuesta-50} o menos'
                                }, 
            {'role': 'user',   'content': f'{consulta_usuario}'},
            ]

        respuesta_sobre_sql = enviar_promt_chat_completions_mode(
            mensaje=mi_prompt, 
            probabilidad_acumulada=1, 
            aleatoriedad=0.5)
        
        self.historico.actualizar_historico(mensaje=respuesta_sobre_sql, role= 'agent', tipo= 'respuesta_informe')
        self.historico_completo.actualizar_historico(mensaje=respuesta_sobre_sql, role= 'agent', tipo= 'respuesta_informe')
        self.historico_completo.guardar_consulta_estructurada(
            usuario= consulta_usuario, 
            codigo_sql=codigo_sql, 
            tabla= tabla_texto, 
            respuesta_llm=respuesta_sobre_sql
        )

        return respuesta_sobre_sql


    # intentar resolver el error en la consulta 
    def resolver_error_SQL(self, consulta_nlp:str, codigo_sql_conflictivo:str, mensaje_error:str): 
        '''Esta función recibe la consulta previa y el error que ha dado al ejecutar el codigo extraido, normalmente los errores
           seran por parte de la ejecución del codigo SQL, producidos errores de compatibilidad con postgreSQL y/o alchemySQL
        '''
        
        mi_prompt = [
            {'role': 'system', 
                      'content': f'Eres un asistente experto en bases de datos, que convierte peticiones de lenguaje natural a código SQL.\
                                   Se te ha hecho una consulta previamente y ha dado el siguiente error \n{mensaje_error}\nEl código de la \
                                   consulta SQL que ha dado error:\n{codigo_sql_conflictivo}\nDEBES DEVOLVER UN NUEVO CÓDIGO SQL, QUE NO DÉ ERROR. \
                                   AQUI TIENES LOS METADATOS DE LA BASE DE DATOS: \n{self.metadata_str}\n. Para expresiones regulares utiliza \
                                   "~" en lugar de "REGEXP".'},      
            {'role': 'user',   'content': f'Mi consulta previa:\n{consulta_nlp}'},
            {'role': 'assistant', 'content': 'DEVUELVES SOLO EL CODIGO SQL SIN NINGUN COMENTARIO EN ESTE FORMATO: ```sql\nEL CODIGO SQL;\n```'}
            ]

        nueva_respuesta_sql = enviar_promt_chat_completions_mode(
            mensaje=mi_prompt, 
            probabilidad_acumulada=1, 
            aleatoriedad=0)
        
        # extrae el codigo sql de la respuesta y se pasa a tabla, tambien se gestionan los errores
        mensaje_o_tabla, codigo_sql, error = self._extract_sql_and_data(respuesta_sql=nueva_respuesta_sql)
            
        return mensaje_o_tabla, codigo_sql, error


    # decidimos si el usuario quiere continuar conversando
    def continuar_conversando(self, usuario, codigo_sql_ejecutado:str, tabla_consulta_anterior = None, 
                              max_tokens_historico:int= 1000):
        '''Clasifica el prompt del usuario de dos maneras: 
            1.- Si este quiere seguir conversando 'continuar' = True o False.
            2.- Si se trata de una nueva consulta o si es una consulta anterior 'nueva_consulta' = True o False'''
        
        # creamos la ventana de historico
        ultimo_historico = self.historico.ventana_historico(max_tokens= max_tokens_historico)
        
        # si NO se ha proporcionado una tabla sobre los resultados
        if tabla_consulta_anterior == None:
            prompt_para_extraer_argumentos = [

                {'role': 'system', 
                        'content': f'Tu objetivo es extraer los argumentos ("continuar" y "nueva_consulta") para\
                        ejecutar la función que te he pasado en tools. Primero debes diferenciar si el usuario quiere\
                        seguir conversando o no: True o False. Segundo debes diferenciar si el usuario tiene una nueva \
                        consulta o no: True o False, sobre la consulta anterior. Para ello aqui tienes el historico \n{ultimo_historico} \
                        y aqui el codigo SQL que se ha ejecutado en la consulta anterior\n{codigo_sql_ejecutado}'
                        },

                {'role': 'user', 'content': f'{usuario}'} ]

            argumentos_extraidos_del_llm = enviar_promt_chat_completions_mode(
                    mensaje=prompt_para_extraer_argumentos, 
                    funciones= SQLAgent.tools, 
                    forzar_funciones= {"type": "function", "function": {"name": "continuar_conversacion"}}, 
                    aleatoriedad= 0, 
                    probabilidad_acumulada=1)

        # si SI se ha proporcionado una tabla 
        else:
            prompt_para_extraer_argumentos = [
                {'role': 'system', 
                        'content': f'Tu objetivo es extraer los argumentos ("continuar" y "nueva_consulta") para\
                        ejecutar la función que te he pasado en tools. Primero debes diferenciar si el usuario quiere\
                        seguir conversando o NO sobre la BBDD: True o False. Segundo debes diferenciar si el usuario tiene una nueva \
                        consulta o no: True o False, sobre la consulta anterior. Para ello aqui tienes el historico \n{ultimo_historico}\n \
                        de la conversación, y la tabla correspondiente al codigo de la consulta:\n{tabla_consulta_anterior}\n\
                        La tabla viene dada por el siguiente codigo SQL:\n{codigo_sql_ejecutado}\n' },

                {'role': 'user', 'content': f'{usuario}'}]

            argumentos_extraidos_del_llm = enviar_promt_chat_completions_mode(
                    mensaje=prompt_para_extraer_argumentos, 
                    funciones= SQLAgent.tools, 
                    forzar_funciones= {"type": "function", "function": {"name": "continuar_conversacion"}}, 
                    aleatoriedad= 0, 
                    probabilidad_acumulada=1)

        return argumentos_extraidos_del_llm['continuar'], argumentos_extraidos_del_llm['nueva_consulta']

    # resolvemos duda sobre consulta anterior
    def pregunta_sobre_consulta_anterior(self, 
        usuario:str, tabla_consulta_anterior:str= None, consulta_sql_anterior:str= None,
        max_tokens_historico:int= 1000, max_tokens_respuesta:int= 1000):

        '''
        Recibe una pregunta del usuario sobre una respuesta anterior del sistema y el agente devuelve una respuesta teniendo \
        acceso al historico y a la los datos de lectura, descripción y predicciones
        '''

        ultimo_historico = self.historico.ventana_historico(max_tokens= max_tokens_historico)

        # actualizamos el historico con la duda del usuario sobre la consulta anterior
        self.historico.actualizar_historico(mensaje= usuario, role= 'user')
        self.historico_completo.actualizar_historico(mensaje= usuario, role= 'user')

        # si no se ha proporcionado la tabla de la consulta anterior
        if tabla_consulta_anterior== None: 
            prompt_conversacion = [
                {'role': 'system', 
                'content': f'Eres un asistente que responde de manera breve pero precisa sobre consultas a bases de datos previas. Para ello, \
                            el codigo SQL de la consulta anterior:\n{consulta_sql_anterior}\n el historico de la conversación:\n{ultimo_historico}\n\
                            Tu respuesta debe ser como maximo de {max_tokens_respuesta-100} tokens o menos'}, 
                                            
                {'role': 'user', 
                'content': f'Responde a mi consulta: \n{usuario}'}
                ]
                
            respuesta_consulta_anterior = enviar_promt_chat_completions_mode(
                        mensaje= prompt_conversacion, 
                        modelo="gpt-4-1106-preview", 
                        maximo_tokens=max_tokens_respuesta, 
                        aleatoriedad=0.2, 
                        probabilidad_acumulada=0.8)

        # Si se ha proporcionado la tabla consulta anterior
        else: 
            prompt_conversacion = [
                {'role': 'system', 
                'content': f'Eres un asistente que responde de manera breve pero precisa sobre consultas a bases de datos previas. Para ello, \
                            El codigo SQL de la consulta anterior:\n{consulta_sql_anterior}\n \
                            el historico de la conversación:\n{ultimo_historico}\n y la tabla de la consulta anterior \n{tabla_consulta_anterior}\n.\
                            Tu respuesta debe ser como maximo de {max_tokens_respuesta-100} tokens o menos'}, 
                                            
                {'role': 'user', 
                'content': f'Responde a mi consulta: \n{usuario}'}
                ]
                
            respuesta_consulta_anterior = enviar_promt_chat_completions_mode(
                        mensaje= prompt_conversacion, 
                        modelo="gpt-4-1106-preview", 
                        maximo_tokens=max_tokens_respuesta, 
                        aleatoriedad=0.2, 
                        probabilidad_acumulada=0.8
                        )
        
        # actualizamos el historico con la respuesta del llm sobre la consutla anterior
        self.historico.actualizar_historico(mensaje=respuesta_consulta_anterior, role= 'agent', tipo= 'respuesta_duda')
        self.historico_completo.actualizar_historico(mensaje=respuesta_consulta_anterior, role= 'agent', tipo= 'respuesta_duda')
        self.historico_completo.guardar_consulta_estructurada(
            usuario= usuario, 
            tabla= tabla_consulta_anterior, 
            respuesta_llm=respuesta_consulta_anterior, 
            codigo_sql=consulta_sql_anterior
        )

        return respuesta_consulta_anterior
