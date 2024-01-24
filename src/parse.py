def parse_proba_predictions(gm_predictions, umbral= 0.99):
    '''Esta función recibe como argumento, unas predicciones en probabilidades y asigna la clase 1 o -1. 
       si no hay una probabilidad maxima superior al umbral, es un outlier -1, si no, inlier 1'''

    gm_labels = []
    for instance in gm_predictions: 
        if max(instance) <umbral: gm_labels.append(-1)
        else:                   gm_labels.append(1)
    return gm_labels

def parse_class_predictions(class_predictions): 
    '''Esta función recibe como argumento, unas predicciones en diferentes clases -1, 0, 1, 2, ... (creadas por DBSCAN) 
       donde haya un -1 la etiqueta se mantiene (outlier o ruido), donde haya otra cosa se establece 1 (inlier o NO ruido)'''
    
    labels = []
    for instance in class_predictions: 
        if instance == -1: labels.append(-1)
        else:              labels.append(1)
    return labels