
import tiktoken

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    '''
    Esta función toma una cadena de texto y un nombre de
    codificación (con un valor predeterminado) y devuelve 
    la cantidad de tokens que esa cadena representaría 
    cuando se codifique con esa codificación específica.
    '''

    encoding = tiktoken.get_encoding(encoding_name)     # Obtener la codificación correspondiente al nombre dado.
    num_tokens = len(encoding.encode(string))           # Codificar la cadena y contar el número de tokens resultantes.

    return num_tokens                                   # Devolver el recuento de tokens.


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""

    # Intentar obtener la codificación para el modelo dado, si no se encuentra, se usa una por defecto.
    try: 
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    # Establecer la cantidad de tokens por mensaje y por nombre según el modelo.
    if model in {
            # Lista de modelos conocidos y su tokenización.
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613"
            }:
        tokens_per_message = 3
        tokens_per_name = 1

    # Manejo de modelos no específicamente listados pero que siguen un patrón conocido.
    elif model == "gpt-3.5-turbo-0301":

        # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_message = 4

        # if there's a name, the role is omitted
        tokens_per_name = -1

    elif "gpt-3.5-turbo" in model:

        print("Warning: gpt-3.5-turbo may update over time. \
        Returning num tokens assuming gpt-3.5-turbo-0613.")

        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")

    elif "gpt-4" in model:

        print("Warning: gpt-4 may update over time. \
        Returning num tokens assuming gpt-4-0613.")

        return num_tokens_from_messages(messages, model="gpt-4-0613")

    # Si el modelo no es reconocido, lanzar un error.
    else:

        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented
            for model {model}.
            See https://github.com/openai/openai-python/blob/main/chatml.md
            for information on how messages are converted to tokens."""
        )

    num_tokens = 0
    for message in messages:

        # Añadir tokens por mensaje.
        num_tokens += tokens_per_message
        for key, value in message.items():

            num_tokens += len(encoding.encode(value))   # Codificar cada parte del mensaje y añadir al recuento de tokens

            if key == "name":
                num_tokens += tokens_per_name           # Ajustar el recuento de tokens si hay un nombre en el mensaje.

    num_tokens += 3                                     # every reply is primed with <|start|>assistant<|message|>

    return num_tokens                                   # Devolver el recuento total de tokens.