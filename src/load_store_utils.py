import os
import joblib
import pandas as pd 
import numpy as np 

def guardar_modelo(modelo, name:str, ruta_modelo:str): 
    '''Esta función guarda un modelo de ML pero primero comprueba si esta o
    en el correspondiente directorio'''

    ruta_completa = os.path.join(ruta_modelo, name)
    if os.path.exists(ruta_completa): 
        print(f'El archivo {name} ya existe en el directorio {ruta_modelo}.')
        return False

    else: 
        joblib.dump(modelo, ruta_completa)
        print(f"Modelo guardado con éxito en {ruta_completa}")
        return True


def cargar_modelo(name, ruta_modelo): 
    '''Esta funciónn carga un modelo, pero primero comprueba si existe el modelo'''

    ruta_completa = os.path.join(ruta_modelo, name)

    if not os.path.exists(ruta_completa):
        print(f"El archivo '{name}' no existe en el directorio '{ruta_modelo}'. No se pudo cargar el modelo.")
        return None

    modelo = joblib.load(ruta_completa)
    print(f"Modelo cargado con éxito desde '{ruta_completa}'")
    return modelo


def guardar_datos_procesados(data, name:str, ruta_datos:str): 
    '''Esta función guarda datos preprocesados, pero primero comprueba si esta o
    no el archivo ya creado.'''

    ruta_completa = os.path.join(ruta_datos, name)
    if os.path.exists(ruta_completa): 
        print(f'Los datos {name} ya existen en el directorio {ruta_datos}.')
        return False

    else: 
        data.to_csv(ruta_completa)
        print(f"Los datos se guardaron con éxito en {ruta_completa}")
        return True


def cargar_lote_datos(m_samples, name, ruta_datos, formato= 'DataFrame'): 
    '''Carga un lote de m ejemplos de un archivo csv determinado de manera aleatoria, 
       en formato dataframe o en numpy array'''

    ruta_completa = os.path.join(ruta_datos, name)

    lote = pd.read_csv(ruta_completa).set_index('Timestamp').iloc[-m_samples:, ::]

    if formato=='DataFrame': return lote
    else:                    return lote.to_numpy()
