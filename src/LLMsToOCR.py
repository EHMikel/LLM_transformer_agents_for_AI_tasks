import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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

from random import choice

import easyocr
from PIL import Image


class LLMsToOCR():

    def __init__(self, 
                 data_file= 'C:/Users/plane/OneDrive/Escritorio/COMPUTING SCIENCE/TFM_transformer_agents/data/signature_detection/test_data.pickle', 
                 documents_path = 'C:/Users/plane/OneDrive/Escritorio/COMPUTING SCIENCE/TFM_transformer_agents/proyecto_yolo/images/test/',
                 model_path= "C:/Users/plane/OneDrive/Escritorio/COMPUTING SCIENCE/TFM_transformer_agents/runs\detect/train\weights/best.pt"):

        self.data_file = data_file                                      # el archivo con los datos
        self.model_path = model_path                                    # el paht al modelo YOLO
        self.data = pd.read_pickle(self.data_file)                      # se carga en dataframe los datos
        self.documents_path = documents_path                            # el path a las imagenes
        self.documents_names = list(self.data['file_name'].unique())    # se extrae la lista de imagenes unicas
        self.detection_model = YOLO(self.model_path)                    # se define el modelo de detección
        

    def choose_document(self): 
        self.docu= choice(self.documents_names)                         # se elige un documento al azar
        self.documents_names.remove(self.docu)                          # se elimina ese documento de la lista
        self.document = os.path.join(self.documents_path, self.docu)    # en esta variable se guarda el full path de la imagen
        return self.docu                                           # aunqeu se gestiona de manera interna se devuelve el documento elegido

    def show_document(self): 
        imagen = cv2.imread(self.document, cv2.IMREAD_GRAYSCALE)  # Cargar en escala de grises
        plt.figure(figsize=(10, 13))
        plt.imshow(imagen, cmap='gray')  # Asegurarse de usar el mapa de colores 'gray'
        plt.axis('off')                  # Desactiva los ejes
        plt.show()

    def extract_text_from_docu(self): 
        imagen =  cv2.imread(self.document)
        img_cv_gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        reader = easyocr.Reader(['en'])       #  objeto lector en ingles
        text_in_image = reader.readtext(img_cv_gray)
        self.text_from_img = ' '.join([detected_text[1] for detected_text 
                                       in text_in_image])
        return self.text_from_img

    def signature_detection(self, show= True):

        # se hacen las predicciones
        self.predictions = self.detection_model.predict(self.document)  

        if show== True: self.predictions[0].show()    # se muestra la imagen con las predicciones

        # se extreae los datos de inferencia
        boxes = self.predictions[0].boxes             # los datos de las bboxes
        predicted_clases = boxes.cls.tolist()         # las clases predichas
        class_names = self.predictions[0].names       # los nombres de las clases
        predicted_conf = boxes.conf.tolist()          # las confianzas de las predicciones
        boxes_ubications = boxes.xywh.tolist()        # las coordenadas de la bboxes
        self.img_shape = boxes.orig_shape             # el tamaño original de la imagen o documento

        # se parsean algunos datos
        predicted_class_w_names = [class_names[int(pred)] for pred in predicted_clases]
        boxes_xy_loc = [location[:2] for location in boxes_ubications]

        predictions_dict = {
            'objetos_detectados': predicted_class_w_names, 
            'confianza': predicted_conf, 
            'ubicación_xy': boxes_xy_loc
            }

        self.predictions_df = pd.DataFrame(predictions_dict)

        return self.predictions_df
    

class OCRAgent(LLMsToOCR): 

    tools = [
        {
        "type": "function",
        "function": {
            "name": "continuar_conversacion",
            "description": 'Extrae los argumentos booleanos "continuar" y "nueva_consulta".\n\
            Si el usuario pregunta algo que no tiene nada que ver con posibles fechas, firmas, iniciales, redacciones detectadas con YOLO\
            ,o sobre el texto extraido de un documento mediante técnicas de OCR, entonces "continuar" = "False"\
            Si puedes responder al prompt del usuario con la información del historico, o demás info: "continuar"= "True", y "nueva_consulta" = "False"\
            Si el usuario hace una consulta que no puedes responder con el historico o demás información, y pide que se carge, procese o que resumas algún\
            otro documento, entonces "continuar"= "True", y "nueva_consulta" = "True". Si "continuar" =  "False" entonces siempre "nueva_consulta" = "False"',

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
                 data_file= 'C:/Users/plane/OneDrive/Escritorio/COMPUTING SCIENCE/TFM_transformer_agents/data/signature_detection/test_data.pickle', 
                 documents_path = 'C:/Users/plane/OneDrive/Escritorio/COMPUTING SCIENCE/TFM_transformer_agents/proyecto_yolo/images/test/',
                 model_path= "C:/Users/plane/OneDrive/Escritorio/COMPUTING SCIENCE/TFM_transformer_agents/runs\detect/train\weights/best.pt", 
                 llm_model= "gpt-4-1106-preview"):
        
        super().__init__(data_file, documents_path, model_path)
        self.historico_completo = HistoricoConversacion()  # incluye las tablas 
        self.historico = HistoricoConversacion()           # solo incluye las respuestas en lenguaje natural y las consultas
        self.llm_model = llm_model


    def informe_resultado(self, consulta_usuario, max_tokens_respuesta:int= 2000): 

        pred_df = self.predictions_df.to_markdown()

        self.historico.actualizar_historico(mensaje=consulta_usuario, role= 'user')
        self.historico_completo.actualizar_historico(mensaje=consulta_usuario, role= 'user')
        
        
        prompt = [
            {'role': 'system', 
                    'content': f'Eres un asistente que combina por un lado, OCR para extraer texto de documentos de una sola pagina en imágenes,\
                                y por otro lado, detecciones de firmas, iniciales, redacciones y fechas escritas a mano, y detectadas mediante\
                                el modelo YOLO. Tu objetivo es dar una idea básica o una descripción breve de lo que es el documento en base al texto extraido \
                                meidante OCR, y debes informar al usuario de cuantas y cuales instancias a predicho YOLO en la imagen, y la  \
                                ubicación aproximada de esas predicciones, pero no expongas las coordenadas de las detecciones explicitamente.\
                                Para ello, el usuario te proporcionará el texto extraido de la imagen que esta en ingles, pero tu debes informar al usuario en castellano\
                                y se te proporciona un dataframe con la info de las predicciones. Adecúa tu respuesta a un maximo de {max_tokens_respuesta-100} tokens o más breve'}, 
                                        
            {'role': 'user', 
                    'content': f'En base al siguiente texto extraido meidante OCR:\n{self.text_from_img}\n, y los resultados hechos por YOLO:\n{pred_df}\n\
                                teniendo en cuenta que la imagen es de tamaño {self.img_shape}, responde a mi consulta:{consulta_usuario}'},

            {'role': 'assistant', 
                    'content': 'Descripción de doumento:\n\n\
                                AQUI HABLAS DEL TEXTO EXTRAIDO POR OCR\n\n\
                                información especial detectada:\n\n\
                                AQUI DAS UNA DESCRIPCIÓN DE LAS PREDICCIONES QUE HA HECHO EL MODELO DE DETECCIÓN. por ejemplo:\
                                - 2 firmas (signatures) con niveles de confianza 91.3% y 77.6%, una ubicada arriba a la derecha de la imagen y las otra en la esquina inferior\n\
                                - 2 redacciones (redactions) con niveles de confianza de 76.8% y 75.8%, ubicadas en la parte central de la imagen.\n\
                                - 1 inicial con nivel de confianza de 30.4%, ubicada arriba a la derecha\n\
                                \ ERES libre de darle más variabilidad al informe'}
                    ]
        # simular_respuesta_generativa(f'El numero total de tokens de tu prompt es: {num_tokens_from_messages(primer_prompt)}, \n\n')

        respuesta = enviar_promt_chat_completions_mode(
                        mensaje= prompt, 
                        modelo=self.llm_model, 
                        maximo_tokens=max_tokens_respuesta, 
                        aleatoriedad=0.5, 
                        probabilidad_acumulada=0.7)
        
        self.historico.actualizar_historico(mensaje=respuesta, role= 'agent', tipo= 'respuesta_informe')
        

        self.historico_completo.actualizar_historico(mensaje= self.text_from_img, role= 'agent', tipo= 'Texto_extraido')
        self.historico_completo.actualizar_historico(mensaje= pred_df, role= 'agent', tipo= 'predicciones_YOLO')
        self.historico_completo.actualizar_historico(mensaje=respuesta, role= 'agent', tipo= 'respuesta_informe')
        self.historico_completo.guardar_consulta_estructurada(
            usuario= consulta_usuario,
            tabla= self.text_from_img, 
            respuesta_llm=respuesta,
            codigo_sql= pred_df         # que no es codigo sql pero se guarda por que son las predicciones
        )

        return respuesta 
    

    def continuar_conversacion(self, usuario:str, max_tokens_historico:int= 3000)-> bool:
        '''Clasifica el prompt del usuario de dos maneras: 
        1.- Si este quiere seguir conversando 'continuar' = True o False.
        2.- Si se trata de una nueva consulta o si es una consulta anterior 'nueva_consulta' = True o False
        '''

        ultimo_historico = self.historico.ventana_historico(max_tokens= max_tokens_historico)
        pred_df = self.predictions_df.to_markdown

        extraccion_argumentos = [
                {'role': 'system', 
                        'content': 
                        f'Eres un asistente que combina por un lado, OCR para extraer texto de documentos de una sola pagina en imágenes,\
                        y por otro lado, detecciones de firmas, iniciales, redacciones y fechas escritas a mano, y detectadas mediante\
                        el modelo YOLO. En este caso Tu objetivo es extraer los argumentos ("continuar" y "nueva_consulta") para\
                        ejecutar la función que te he pasado en tools. Primero debes diferenciar si el usuario quiere\
                        seguir conversando o no: True o False. Segundo debes diferenciar si el usuario tiene una nueva \
                        consulta o no: True o False, sobre la consulta anterior. Para ello aqui tienes el historico:\n{ultimo_historico}\n\
                        Y ten en cuenta la informacion que te ha pasado el usuario. Si puedes responder la pregunta del usuario con la \
                        información que tienes "nueva_consulta= False" de lo contrario "nueva_consulta"= True'
                        },

                {'role': 'user', 
                    'content': f'A partir del siguiente texto extraído:\n {self.text_from_img}\n, de la imagen con tamaño: \n {self.img_shape}\n y \
                                 la siguiente predicciones hechas por YOLO: \n {pred_df}\n responde a mi consulta: \n{usuario}'}
            ]

        argumentos_extraidos_del_llm = enviar_promt_chat_completions_mode(
                mensaje=extraccion_argumentos, 
                funciones= OCRAgent.tools, 
                forzar_funciones= {"type": "function", "function": {"name": "continuar_conversacion"}}, 
                aleatoriedad= 0, 
                probabilidad_acumulada=1)
        
        return argumentos_extraidos_del_llm['continuar'], argumentos_extraidos_del_llm['nueva_consulta']
    
    def pregunta_sobre_consulta_anterior(self, usuario:str, max_tokens_respuesta:int= 2000, max_tokens_historico:int= 3000): 
        
        '''Recibe una pregunta del usuario sobre una respuesta anterior del sistema y el agente devuelve una respuesta teniendo \
        acceso al historico y a la los datos de lectura, descripción y predicciones'''

        ultimo_historico = self.historico.ventana_historico(max_tokens= max_tokens_historico)
        pred_df = self.predictions_df.to_markdown()

        self.historico.actualizar_historico(mensaje= usuario, role= 'user')
        self.historico_completo.actualizar_historico(mensaje= usuario, role= 'user')

        prompt_conversacion = [
        {'role': 'system', 'content': 
                        f'Eres un asistente que combina por un lado, OCR para extraer texto de documentos de una sola pagina en imágenes,\
                        y por otro lado, detecciones de firmas, iniciales, redacciones y fechas escritas a mano, y detectadas mediante\
                        el modelo YOLO. El historico de la conversación: {ultimo_historico}\
                        Tu respuesta debe ser como maximo de {max_tokens_respuesta-100} tokens'}, 
                                    
        {'role': 'user', 
                    'content': f'A partir del siguiente texto extraído:\n {self.text_from_img}\n, de la imagen con tamaño: \n {self.img_shape}\n y \
                                 la siguiente predicciones hechas por YOLO: \n {pred_df}\n responde a mi consulta: \n{usuario}'}
                                    
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