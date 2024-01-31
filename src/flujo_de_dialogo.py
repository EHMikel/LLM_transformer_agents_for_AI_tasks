from calcular_tokens import num_tokens_from_string
from open_ai_utils import enviar_promt_chat_completions_mode


def ventana_de_historico(historico_completo:str, max_tokens:int= 1000)->str:
    """
    Procesa el histórico de la conversación y acumula mensajes hasta alcanzar un límite de tokens, para devolver una ventana de historico
    """
    # Separar el histórico en palabras.
    palabras = historico_completo.split(' ')[::-1]             # guardamos el historico de más reciente a más antiguo en una lista 
    historico_limitado = []                                    # variable de ventana de historico
    tokens_acumulados = 0                                      # variable de tokens acumulados

    for palabra in palabras:                                       # para cada palabra dentro del historico
        tokens_palabra = num_tokens_from_string(palabra)           # Calcular los tokens para la palabra actual.
        if tokens_acumulados + tokens_palabra > max_tokens: break  # si se excede el límite de tokens rompemos el bucle

        else:                                       # si no se excede el limite de tokens ...
            historico_limitado.append(palabra)      # Agregar la palabra al histórico limitado 
            tokens_acumulados += tokens_palabra     # aumenta la cantidad de tokens acumulados

    historico_limitado_str = ' '.join(historico_limitado[::-1]) # ahora unimos en str el historico en el orden original separado por un espacio
    return historico_limitado_str                               # devolvemos la ventana de historico


def guardar_historico(historico:str, mensaje_usuario:str, respuesta_sistema:str, contador_interacciones:int): 
    '''Recibe el historico anterior y la nueva interaccion entre el usuario y el agente, y actualiza el historico. \
        Además, sube el valor del la variable global contador_interacciones y lo devulve junto con el historico.'''
    
    historico += f'prompt USUARIO_{contador_interacciones}:\n{mensaje_usuario}\n\n'
    historico += f'Respuesta AGENTE_{contador_interacciones}:\n{respuesta_sistema}\n\n'
    contador_interacciones += 1
    return historico, contador_interacciones


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