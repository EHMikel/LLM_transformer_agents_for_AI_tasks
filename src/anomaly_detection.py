import pandas as pd 
import numpy as np 
from load_store_utils import cargar_modelo
from calcular_tokens import num_tokens_from_messages, num_tokens_from_string
from open_ai_utils import enviar_promt_chat_completions_mode, simular_respuesta_generativa
import os



class AnomalyDetection(): 

    # data_path = '../data/Anomaly_detection/processed_data/'
    # data_file = 'HVAC_test_processeded.csv'
    # models_path = './models/anomaly_detection/'
    # full_data_path = os.path.join(data_path, data_file)

    columnas_base = ['T_Supply', 'T_Return', 'SP_Return', 'T_Saturation', 'T_Outdoor',
                 'RH_Supply', 'RH_Return', 'RH_Outdoor', 'Energy', 'Power']
    
    # los modelos 
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
                 models_path = './models/anomaly_detection/'):
        
        self.data_path = data_path
        self.data_file = data_file
        self.full_data_path = os.path.join(data_path, data_file)
        self.data = pd.read_csv(self.full_data_path)
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


    def lectura_de_datos(self, m_samples, formato= 'DataFrame'): 
        '''Carga un lote de los últimos m datos, devuleve los datos originales y los escalados
        '''

        self.lectura_datos = self.data.set_index('Timestamp').iloc[-m_samples:, ::]
        self.lectura_datos_scl = pd.DataFrame(
            self.scaler.transform(self.lectura_datos), 
            index= self.lectura_datos.index, 
            columns= self.lectura_datos.columns
            ) 

        # if formato=='DataFrame': return self.lectura_datos, self.lectura_datos_scl
        # else:                    return self.lectura_datos.to_numpy(), self.lectura_datos_scl.to_numpy()
            

    @staticmethod
    def parse_proba_predictions(gm_predictions, umbral= 0.99):
        '''Esta función recibe como argumento, unas predicciones en probabilidades y asigna la clase 1 o -1. 
        si no hay una probabilidad maxima superior al umbral, es un outlier -1, si no, inlier 1'''

        gm_labels = []
        for instance in gm_predictions: 
            if max(instance) <umbral: gm_labels.append(-1)
            else:                   gm_labels.append(1)
        return gm_labels

    @staticmethod
    def parse_class_predictions(class_predictions): 
        '''Esta función recibe como argumento, unas predicciones en diferentes clases -1, 0, 1, 2, ... (creadas por DBSCAN) 
        donde haya un -1 la etiqueta se mantiene (outlier o ruido), donde haya otra cosa se establece 1 (inlier o NO ruido)'''
        
        labels = []
        for instance in class_predictions: 
            if instance == -1: labels.append(-1)
            else:              labels.append(1)
        return labels

    @staticmethod
    def mayority_voting(row): 
        count_outlier_predictions = (row == -1).sum()
        if count_outlier_predictions > (len(row) / 2): return 1  # anomaly
        else:                                          return 0  # not an anomaly

    

    def predict_anomalies(self, lectura_datos):
        '''Esta función realiza unas predicciones con unos modelos cargados previamente
           y devuelve un DataFrame con las predicciones
        '''

        scaled_data = self.scaler.transform(lectura_datos)       
        dct_models = dict(zip(self.model_names, self.models))

        models_and_predictions = {}
        for name, model in dct_models.items():
            
            if name == 'Gaussian_Mixtures': 
                pred = model.predict_proba(scaled_data)
                predictions = AnomalyDetection.parse_proba_predictions(pred)
            elif name == 'KNN_DBSCAN':
                pred = model.predict(scaled_data)   
                predictions = AnomalyDetection.parse_class_predictions(pred)
            else: 
                predictions = model.predict(scaled_data)

            models_and_predictions[name] = predictions
        
        predictions = pd.DataFrame(models_and_predictions, index= scaled_data.index)
        predictions['anomaly'] = predictions.apply(lambda x: AnomalyDetection.mayority_voting(x), axis= 1)

        return predictions

    # def cargar_modelos(self): 

    #     nombre_scl = 'scaler_trained.pickle'        # el escalador de datos

    #     # los modelos 
    #     nombre_if = 'IF_trained.pickle'             # isolation forest
    #     nombre_ee = 'EE_trained.pickle'             # eliptic envelope
    #     nombre_lof = 'LOF_trained.pickle'           # local outlier factor
    #     nombre_ocsvm = 'OCSVM_trained.pickle'       # one class svm
    #     nombre_gm = 'GM_trained.pickle'             # gaussian mixtures
    #     nombre_knn = 'dbscan_knn.pickle'            # dbscan_knn

    #     # cargamos los modelos
    #     self.scaler = cargar_modelo(name= nombre_scl, ruta_modelo=self.models_path)
    #     self.IF = cargar_modelo(name=nombre_if, ruta_modelo=self.models_path)
    #     self.EE = cargar_modelo(name=nombre_ee, ruta_modelo=self.models_path)
    #     self.LOF = cargar_modelo(name=nombre_lof, ruta_modelo=self.models_path)
    #     self.OCSVM = cargar_modelo(name= nombre_ocsvm, ruta_modelo= self.models_path)
    #     self.GM = cargar_modelo(name= nombre_gm, ruta_modelo= self.models_path)
    #     self.KNN_dbscan = cargar_modelo(name= nombre_knn, ruta_modelo= self.models_path)

    #     self.models = [self.IF, self.EE, self.LOF, self.OCSVM, self.GM, self.KNN_dbscan]

    #     return self.scaler, self.models

class AnomalyDetectionAgent(AnomalyDetection):

    tools = [
        {
            "type": "function",
            "function": {
                "name": "cargar_lote_datos",
                "description": "Carga un lote de datos de un archivo CSV basado en el número de muestras especificadas cada muestra representa 1 hora.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "m_samples": {
                            "type": "integer",
                            "description": "Número de muestras a cargar, si no se proporciona será por defecto 5"
                        },
                        "formato" : {
                            "type": "string",
                            "enum": ["DataFrame", "numpy"],
                            "description": "Formato en el que se devuelven los datos, DataFrame por defecto"
                        }
                    },
                    "required": ["m_samples"]
                        }
                    }
                }
            ]
    
    def nueva_consulta_anomaly_detection(consulta_usuario, datos_base, descr_datos, predicciones, max_tokens_respuesta:int= 500): 

        primer_prompt = [
        {'role': 'system', 
                'content': f'Eres un asistente de ayuda para un diagnostico de fallos en un sistema energético que genera respuestas concisas. Primero \
                            generas un informe resumido y breve de lecturas; con número de registros, el maximo y minimo, media, etc. de las variables \
                            energy, power, T_outdoor, T_supply, RH_outdoor. Siempre das las unidades de las variables, pero no hables de cuales son  \
                            las variables de lectura, el usuario ya lo sabe.  Después, haces un breve resumen de predicciones: cuantas y cuales de las \
                            instancias se consideran anomalas según las predicciones; 0 significa lectura normal, 1 anomalía. \
                            Adecúa tu respuesta a un maximo de {max_tokens_respuesta-100} tokens o más breve'}, 
                                    
        {'role': 'user', 
                'content': f'A partir de la siguiente lectura:\n {datos_base}\n, con descripción: \n {descr_datos}\n y la siguiente \
                            predicción: \n {predicciones}\n responde a mi consulta: \n{consulta_usuario}'},

        {'role': 'assistant', 
                'content': 'INFORME: \n generas aqui tu informe \n \
                            PREDICCIÓN: \n comentas de manera concisa y breve que lecuras se consideran anomalas y cuales no por ´mayority voting´\
                            COCLUSIÓN: \n haz una breve conclusión'}]
        
        # simular_respuesta_generativa(f'El numero total de tokens de tu prompt es: {num_tokens_from_messages(primer_prompt)}, \n\n')

        respuesta = enviar_promt_chat_completions_mode(
                        mensaje= primer_prompt, 
                        modelo="gpt-4-1106-preview", 
                        maximo_tokens=max_tokens_respuesta, 
                        aleatoriedad=0.1, 
                        probabilidad_acumulada=1)
        
        return respuesta 
    pass
    