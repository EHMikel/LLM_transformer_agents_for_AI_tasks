import pandas as pd 
import numpy as np 
from load_store_utils import cargar_modelo
from calcular_tokens import num_tokens_from_messages, num_tokens_from_string
from open_ai_utils import enviar_promt_chat_completions_mode, simular_respuesta_generativa
from flujo_de_dialogo import HistoricoConversacion
import os
import warnings

# Ignorar todos los warnings
warnings.filterwarnings('ignore')




class AnomalyDetection(): 

    columnas_base = ['T_Supply', 'T_Return', 'SP_Return', 'T_Saturation', 'T_Outdoor',
                 'RH_Supply', 'RH_Return', 'RH_Outdoor', 'Energy', 'Power']
    
    # los archivos de los modelos 
    nombre_if = 'IF_trained.pickle'             # isolation forest
    nombre_ee = 'EE_trained.pickle'             # eliptic envelope
    nombre_lof = 'LOF_trained.pickle'           # local outlier factor
    nombre_ocsvm = 'OCSVM_trained.pickle'       # one class svm
    nombre_gm = 'GM_trained.pickle'             # gaussian mixtures
    nombre_knn = 'dbscan_knn.pickle'            # dbscan_knn
    nombre_scl = 'scaler_trained.pickle'        # el escalador de datos

    def __init__(self, 
                 data_path ='../data/Anomaly_detection/processed_data/', 
                 data_file = 'HVAC_test_processeded.csv', 
                 models_path = './models/anomaly_detection/',
                 ):
        
        self.data_path = data_path
        self.data_file = data_file
        self.full_data_path = os.path.join(data_path, data_file)
        self.models_path = models_path
        self.data = pd.read_csv(self.full_data_path).set_index('Timestamp')
        self.basic_data = self.data[AnomalyDetection.columnas_base]
        self.model_names = ['isolation_forest', 'Eliptic_Envelope', 'Local_Outlier_Factor', 'One_Class_SVM', 'Gaussian_Mixtures', 'KNN_DBSCAN']
        

    def cargar_modelos(self): 
        # cargamos los modelos
        self.scaler = cargar_modelo(name= AnomalyDetection.nombre_scl, ruta_modelo=self.models_path)
        self.IF = cargar_modelo(name=AnomalyDetection.nombre_if, ruta_modelo=self.models_path)
        self.EE = cargar_modelo(name=AnomalyDetection.nombre_ee, ruta_modelo=self.models_path)
        self.LOF = cargar_modelo(name=AnomalyDetection.nombre_lof, ruta_modelo=self.models_path)
        self.OCSVM = cargar_modelo(name= AnomalyDetection.nombre_ocsvm, ruta_modelo= self.models_path)
        self.GM = cargar_modelo(name= AnomalyDetection.nombre_gm, ruta_modelo= self.models_path)
        self.KNN_dbscan = cargar_modelo(name= AnomalyDetection.nombre_knn, ruta_modelo= self.models_path)

        self.models = [self.IF, self.EE, self.LOF, self.OCSVM, self.GM, self.KNN_dbscan]


    def lectura_de_datos(
            self, modo:str= 'ultimas', m_samples= None, 
            fecha_ini= None, fecha_fin= None, horario_ini='18:00:00', horario_fin= '23:30:00'): #formato= 'DataFrame'
        
        '''Carga un lote de  datos, y  almacena los datos originales y los escalados como atributo de insancia, 
           Los modos en los cuales se pueden extraer los datos son ultimas o primeras intancias o entre fechas
        '''

        if modo.lower() == 'ultimas':
            self.lectura_datos = self.data.iloc[-m_samples:, ::]                   # .set_index('Timestamp')
            self.lectura_datos_base = self.basic_data.iloc[-m_samples:, ::]        # .set_index('Timestamp')
            self.lectura_datos_scl = pd.DataFrame(
                self.scaler.transform(self.lectura_datos), 
                index= self.lectura_datos.index, 
                columns= self.lectura_datos.columns
                ) 
        if modo.lower() == 'primeras': 
            self.lectura_datos = self.data.iloc[:m_samples, ::]                   # .set_index('Timestamp')
            self.lectura_datos_base = self.basic_data.iloc[-m_samples:, ::]        # .set_index('Timestamp')
            self.lectura_datos_scl = pd.DataFrame(
                self.scaler.transform(self.lectura_datos), 
                index= self.lectura_datos.index, 
                columns= self.lectura_datos.columns
                )
        if modo.lower() == 'fechas': 

            #horario= ' 00:00:00'
            fecha_ini_str = f"{fecha_ini} {horario_ini}"
            fecha_fin_str = f"{fecha_fin} {horario_fin}"
            try:
                self.lectura_datos = self.data.loc[fecha_ini_str:fecha_fin_str]                   # .set_index('Timestamp')
                self.lectura_datos_base = self.basic_data.loc[fecha_ini_str:fecha_fin_str]         # .set_index('Timestamp')
                #print(self.lectura_datos)
                self.lectura_datos_scl = pd.DataFrame(
                    self.scaler.transform(self.lectura_datos), 
                    index= self.lectura_datos.index, 
                    columns= self.lectura_datos.columns
                    )
            except Exception as e: 
                simular_respuesta_generativa(f'Se ha producido un error al cargar los datos entre fechas:\n{e}')
        # if formato=='DataFrame': return self.lectura_datos, self.lectura_datos_scl
        # else:                    return self.lectura_datos.to_numpy(), self.lectura_datos_scl.to_numpy()
            

        # '''Esta función recibe como argumento, unas predicciones en probabilidades y asigna la clase 1 o -1. 
        # si no hay una probabilidad maxima superior al umbral, es un outlier -1, si no, inlier 1'''

    @staticmethod
    def _parse_proba_predictions(gm_predictions, umbral= 0.99):

        gm_labels = []
        for instance in gm_predictions: 
            if max(instance) <umbral: gm_labels.append(-1)
            else:                     gm_labels.append(1)
        return gm_labels


    @staticmethod
    def _parse_class_predictions(class_predictions): 
        '''Esta función recibe como argumento, unas predicciones en diferentes clases -1, 0, 1, 2, ... (creadas por DBSCAN) 
        donde haya un -1 la etiqueta se mantiene (outlier o ruido), donde haya otra cosa se establece 1 (inlier o NO ruido)'''
        
        labels = []
        for instance in class_predictions: 
            if instance == -1: labels.append(-1)
            else:              labels.append(1)
        return labels

    # frontera dura entre anomalía y no anomalía
    @staticmethod
    def _mayority_voting(row): 
        count_outlier_predictions = (row == -1).sum()
        if count_outlier_predictions > (len(row) / 2): return -1  # anomaly
        else:                                          return 1   # not an anomaly

    # frontera suave entre anomalía y valor tipico
    @staticmethod
    def _rate_anomaly_grade(row): 
        outlier_predictions = (row == -1).sum()
        if outlier_predictions <= (len(row)/3):      return 'lectura común'
        elif outlier_predictions <= (len(row)*0.5):  return 'lectura poco común'
        elif outlier_predictions <= (len(row)*0.75): return 'lectura atípica'
        else:                                        return 'anomalía'

    
    def predict_anomalies(self):
        '''Esta función realiza unas predicciones con unos modelos cargados previamente
           y devuelve un DataFrame con las predicciones
        '''

        #scaled_data = self.scaler.transform(self.lectura_datos)       
        dct_models = dict(zip(self.model_names, self.models))

        models_and_predictions = {}
        for name, model in dct_models.items():
            
            if name == 'Gaussian_Mixtures': 
                pred = model.predict_proba(self.lectura_datos_scl)
                predictions = AnomalyDetection._parse_proba_predictions(pred)
            elif name == 'KNN_DBSCAN':
                pred = model.predict(self.lectura_datos_scl)   
                predictions = AnomalyDetection._parse_class_predictions(pred)
            else: 
                predictions = model.predict(self.lectura_datos_scl)

            models_and_predictions[name] = predictions
        
        predictions = pd.DataFrame(models_and_predictions, index= self.lectura_datos_scl.index)
        predictions['anomaly_grade'] = predictions.apply(lambda x: AnomalyDetection._rate_anomaly_grade(x), axis= 1)
        self.predictions = predictions

        return predictions


class ADAgent(AnomalyDetection):

    tools = [
        {
        "type": "function",
        "function": {
            "name": "lectura_de_datos",
            "description": 'Tu objetivo es extraer 4 argumentos ("modo", "m_samples", "fecha_ini", "fecha_fin"), para una carga de \
            datos con la librería pandas. Cada instancia del dataset corresponden a lecturas en intervalos de 30 minutos. El argumento "modo" determina como \
            se cargaran los datos; si las primeras o ultimas m instancias, o m instancias entre fechas. En caso de que el usuario pida las primeras o \
            ultimas intancias no hacen falta las fechas, ("fecha_ini"="None" y "fecha_fin"="None"), y Debes reconocer cuantas instancias se deben cargar "m_samples" \
            Sin embargo, si el usuario pide una consulta entre fechas, no hace falta extraer un numéro de instancias ("m_samples"= "None") \
            pero debes reconocer las fechas que ha pedido el usuario "fecha_ini" y "fecha_fin", el codigo que se ejecutará si el usuario pide',

            "parameters": {
                "type": "object",
                "properties": {
                    "modo" : {
                        "type": "string",
                        "enum": ["primeras", "ultimas", "fechas"],
                        "description": "El modo en el que se cargaran los datos, si el usuariro no proporciona instrucciones, por defecto será útlimas;\
                         El codigo que se ejecutara dependiendo del modo:\
                        'ultimas': self.lectura_datos = self.data.iloc[-m_samples:, ::]\
                        'primeras': self.lectura_datos = self.data.iloc[m_samples:, ::]\
                        'fechas': self.lectura_datos = self.data.loc[fecha_ini:fecha_fin]"
                            },
                    "m_samples": {
                        "type": "integer",
                        "description": "Número de muestras a cargar, si no se proporciona será por defecto 10. Si modo= 'fechas' entonces m_samples= 'none'"
                            },
                    "fecha_ini" : {
                        "type": "string",
                        "description": "La fecha inicial en caso del que le modo sea 'fechas'. debes extraer la fecha en este formato: yyyy-mm-dd\
                        Si el usuario no especifica una fecha inicial entonces 'fecha_ini' = 2021-04-13\
                        Si 'modo'= 'primeras' o 'ultimas' entonces 'fecha_ini'= None"
                            },
                    "fecha_fin" : {
                        "type": "string",
                        "description": "La fecha final en caso del que le modo sea 'fechas'. debes extraer la fecha en este formato: yyyy-mm-dd\
                        Si el usuario no especifica una fecha final entonces 'fecha_fin' = 2021-04-14\
                        Si 'modo'= 'primeras' o 'ultimas' entonces 'fecha_fin'= None"
                            }
                        },
                "required": ["modo", "m_samples", "fecha_ini", "fecha_fin"]
                            
                        }
                    }
                },
        {
        "type": "function",
        "function": {
            "name": "continuar_conversacion",
            "description": 'Extrae los argumentos booleanos "continuar" y "nueva_consulta".\n\
            Si el usuario pregunta algo que no tiene nada que ver con posibles consultas sobre lecturas de datos o predicciones, entonces "continuar" = "False"  \
            Si puedes responder al prompt del usuario con la información del historico, o demás info: "continuar"= "True", y "nueva_consulta" = "False"\
            Si el usuario hace una consulta que no puedes responder con el historico o demás información, y requiere de una nueva lectura de datos, entonces: \
            "continuar"= "True", y "nueva_consulta" = "True". Si "continuar" =  "False" entonces siempre "nueva_consulta" = "False"\
            Debes diferenciar si el usuario quiere seguir conversando sobre la consulta anterior o tiene una nueva consulta, si el usuario se desvía del \
            tema y escribe un prompt que no tiene nada que ver con el tema actual p.e ¿como se hacen los macarrones? "continuar" =  "False"',
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

     
    def __init__(self, 
                 data_path ='../data/Anomaly_detection/processed_data/', 
                 data_file = 'HVAC_test_processeded.csv', 
                 models_path = './models/anomaly_detection/'):
        super().__init__(data_path, data_file, models_path)

        self.historico_completo = HistoricoConversacion()  # incluye las tablas 
        self.historico = HistoricoConversacion()           # solo incluye las respuestas en lenguaje natural y las consultas

    def map_prompt_to_load_data(self, consulta_usuario:str): 

        extraccion_argumentos = [
        {'role': 'system', 
                 'content': f'Eres un asistente de ayuda para un diagnostico de fallos en un sistema energético. Tu objetivo es \
                 extraer los argumentos necesarios para ejecutar la función que te he pasado en tools, basandote en que El dataset \
                 contiene {len(self.data)} entradas:\nDesde la fecha: {self.data.index.min()}, hasta: {self.data.index.max()}. TEN \
                 EN CUENTA LAS FECHAS DEL DATASET cuando el usuario te pida las lecturas de un día especifico. \
                 PARA TI HOY ES 2021-04-14 23:30:00. Las lecturas del dataset corresponden a lecturas en intervalos de 30 minutos.'}, 
        {'role': 'user', 'content': f'{consulta_usuario}'},
            ]

        argumentos_extraidos_del_llm = enviar_promt_chat_completions_mode(
            mensaje=extraccion_argumentos, 
            funciones= ADAgent.tools, 
            forzar_funciones= {"type": "function", "function": {"name": "lectura_de_datos"}}, 
            aleatoriedad= 0, 
            probabilidad_acumulada=1, 
        )

        # print(argumentos_extraidos_del_llm)
        return argumentos_extraidos_del_llm


    
    def informe_resultado(self, consulta_usuario, max_tokens_respuesta:int= 1000): 

        lectura_datos= self.lectura_datos_base.to_markdown()
        descr_datos = self.lectura_datos_base.describe().to_markdown()
        predictions = self.predictions.to_markdown()

        self.historico.actualizar_historico(mensaje=consulta_usuario, role= 'user')
        self.historico_completo.actualizar_historico(mensaje=consulta_usuario, role= 'user')

        primer_prompt = [
        {'role': 'system', 
                'content': f'Eres un asistente de ayuda para un diagnostico de fallos en un sistema energético que genera respuestas concisas. Primero \
                            generas un informe resumido y breve de lecturas; con número de registros, el maximo y minimo, media, etc. de las variables \
                            energy, power, T_outdoor, T_supply, RH_outdoor. Siempre das las unidades de las variables, pero no hables de cuales son  \
                            las variables de lectura, el usuario ya lo sabe.  Después, haces un breve resumen de predicciones: cuantas y cuales de las \
                            instancias se consideran anomalas según las predicciones; 0 significa lectura normal, 1 anomalía. \
                            Adecúa tu respuesta a un maximo de {max_tokens_respuesta-100} tokens o más breve'}, 
                                    
        {'role': 'user', 
                'content': f'A partir de la siguiente lectura:\n {lectura_datos}\n, con descripción: \n {descr_datos}\n y la siguiente \
                            predicción: \n {predictions}\n responde a mi consulta: \n{consulta_usuario}'},

        {'role': 'assistant', 
                'content': 'INFORME:\ngeneras aqui tu informe \n \
                            PREDICCIÓN:\ncomentas de manera concisa y breve que lecuras se consideran anomalas y cuales no por ´mayority voting´\
                            COCLUSIÓN:\nhaz una breve conclusión'}]
        
        # simular_respuesta_generativa(f'El numero total de tokens de tu prompt es: {num_tokens_from_messages(primer_prompt)}, \n\n')

        respuesta = enviar_promt_chat_completions_mode(
                        mensaje= primer_prompt, 
                        modelo="gpt-4-1106-preview", 
                        maximo_tokens=max_tokens_respuesta, 
                        aleatoriedad=0.1, 
                        probabilidad_acumulada=1)
        
        self.historico.actualizar_historico(mensaje=respuesta, role= 'agent', tipo= 'respuesta_informe')
        

        self.historico_completo.actualizar_historico(mensaje= lectura_datos, role= 'agent', tipo= 'lectura_datos')
        self.historico_completo.actualizar_historico(mensaje= predictions, role= 'agent', tipo= 'predicciones')
        self.historico_completo.actualizar_historico(mensaje=respuesta, role= 'agent', tipo= 'respuesta_informe')
        self.historico_completo.guardar_consulta_estructurada(
            usuario= consulta_usuario,
            tabla= lectura_datos, 
            respuesta_llm=respuesta,
            codigo_sql= predictions         # que no es codigo sql pero se guarda por que son las predicciones, tendre que cambiar el nombre de este argumento
        )

        return respuesta 


    def continuar_conversacion(self, usuario:str, max_tokens_historico:int= 1000)-> bool:
        '''Clasifica el prompt del usuario de dos maneras: 
        1.- Si este quiere seguir conversando 'continuar' = True o False.
        2.- Si se trata de una nueva consulta o si es una consulta anterior 'nueva_consulta' = True o False'''

        ultimo_historico = self.historico.ventana_historico(max_tokens= max_tokens_historico)
        lectura_datos = self.lectura_datos_base.to_markdown()
        descr_datos = self.data.describe().to_markdown()
        predictions = self.predictions.to_markdown()

        extraccion_argumentos = [
                {'role': 'system', 
                        'content': f'Tu objetivo es extraer los argumentos ("continuar" y "nueva_consulta") para\
                        ejecutar la función que te he pasado en tools. Primero debes diferenciar si el usuario quiere\
                        seguir conversando o no: True o False. Segundo debes diferenciar si el usuario tiene una nueva \
                        consulta o no: True o False, sobre la consulta anterior. Para ello aqui tienes el historico:\n{ultimo_historico}\n\
                        Y ten en cuenta la informacion qeu te ha pasado el usuario. Si puedes responder la pregunta del usuario con la \
                        información que tienes "nueva_consulta= False" de lo contrario "nueva_consulta"= True'
                        },

                {'role': 'user', 
                    'content': f'A partir de la siguiente lectura:\n {lectura_datos}\n, con descripción: \n {descr_datos}\n y la siguiente \
                                predicción: \n {predictions}\n responde a mi consulta: \n{usuario}'}
            ]

        argumentos_extraidos_del_llm = enviar_promt_chat_completions_mode(
                mensaje=extraccion_argumentos, 
                funciones= ADAgent.tools, 
                forzar_funciones= {"type": "function", "function": {"name": "continuar_conversacion"}}, 
                aleatoriedad= 0, 
                probabilidad_acumulada=1)
        
        return argumentos_extraidos_del_llm['continuar'], argumentos_extraidos_del_llm['nueva_consulta']


    def pregunta_sobre_consulta_anterior(self, usuario:str, max_tokens_respuesta:int= 1000, max_tokens_historico:int= 1000): 
        
        '''Recibe una pregunta del usuario sobre una respuesta anterior del sistema y el agente devuelve una respuesta teniendo \
        acceso al historico y a la los datos de lectura, descripción y predicciones'''

        ultimo_historico = self.historico.ventana_historico(max_tokens= max_tokens_historico)
        lectura_datos = self.lectura_datos_base.to_markdown()
        descr_datos = self.data.describe().to_markdown()
        predictions = self.predictions.to_markdown()

        self.historico.actualizar_historico(mensaje= usuario, role= 'user')
        self.historico_completo.actualizar_historico(mensaje= usuario, role= 'user')

        prompt_conversacion = [
        {'role': 'system', 'content': f'Eres un asistente de ayuda para un diagnostico de fallos en un sistema energético HVAC que responde de manera concisa sobre un \
                                        informe generado previamente. Siempre pones las unidades de las variables. El historico de la conversación: {ultimo_historico}\
                                        recuerda: 1 significa valor típico y -1 valor atípico. Tu respuesta debe ser como maximo de {max_tokens_respuesta-100} tokens'}, 
                                    
        {'role': 'user', 'content': f'A partir de la siguiente lectura:\n {lectura_datos}\n, con descripción: \n {descr_datos}\n y la siguiente \
                                    predicción: \n {predictions}\n responde a mi consulta: \n{usuario}'}]
        
        respuesta = enviar_promt_chat_completions_mode(
                mensaje= prompt_conversacion, 
                modelo="gpt-4-1106-preview", 
                maximo_tokens=max_tokens_respuesta, 
                aleatoriedad=0.2, 
                probabilidad_acumulada=0.8)
        
        self.historico.actualizar_historico(mensaje=respuesta, role= 'agent', tipo= 'respuesta_consulta_anterior')
        self.historico_completo.actualizar_historico(mensaje=respuesta, role= 'agent', tipo= 'respuesta_consulta_anterior')

        return respuesta
    
    