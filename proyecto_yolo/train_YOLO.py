from ultralytics import YOLO
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import warnings

warnings.filterwarnings('ignore')

def main(): 
      

    # UNA VEZ ENTRENADO A MEDIAS EL MODELO
    # half_trained_model = YOLO('C:/Users/plane/OneDrive/Escritorio/COMPUTING SCIENCE/TFM_transformer_agents/runs/detect/train15/weights/last.pt')

    # Comprueba si CUDA (GPU support) está disponible
    if torch.cuda.is_available():
        print("CUDA (GPU support) está disponible en este sistema.")
        print("Número de dispositivos GPU disponibles:", torch.cuda.device_count())
        print("Nombre del dispositivo GPU actual:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("CUDA (GPU support) no está disponible en este sistema.")

    # Inicia el entrenamiento

    yaml_path = 'C:/Users/plane/OneDrive/Escritorio/COMPUTING SCIENCE/TFM_transformer_agents/proyecto_yolo/signature_detection.yaml'
    #checkpoints_path = 'C:/Users/plane/OneDrive/Escritorio/COMPUTING SCIENCE/TFM_transformer_agents/runs/detect/train15/weights/last.pt'  # Ajusta esta ruta'

    model=YOLO("yolov8m.pt")

    results = model.train(
        data=yaml_path,
        epochs=100,          # Número de épocas 
        batch=-1,            # Tamaño del batch
        imgsz=640,           # Tamaño de la imagen
        optimizer = 'auto',  # YOLO utliza o Adam 
        device= 0,           # entrenar en GPU
        patience= 10,        # monitorización  
        dropout= 0.2,        # regularización del 20 %
        seed= 0,             # se fija la semilla
        verbose= True,       # muestra los resultados
        plots= True          # guarda gráficos 
        )
    
    model.save('sign_detect_yolo.pt')

    # results = half_trained_model.train(data=yaml_path,
    #                       epochs=100,    # Número de épocas de entrenamiento
    #                       batch=8,       # Tamaño del batch
    #                       imgsz=640,     # Tamaño de la imagen
    #                       resume= True,
    #                       device= 0)      # Usa 'cuda' para entrenar en GPU
    # half_trained_model.save('sign_detect_yolo.pt')

if __name__ == '__main__': 
    main()