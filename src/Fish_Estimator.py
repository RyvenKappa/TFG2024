"""
Modulo que contiene funciones para estimar el numero de peces
"""
import pandas as pd
import numpy as np
import ultralytics.engine.results

def estimate_fish_number(data:pd.DataFrame):
    """
    Recibo un array con las cajas, imagen original y el shape original.
    A través de las cajas y el shape original analizaremos cuantos peces hay
    @returns 2 si hay 2
    @returns 1 si hay 1
    """
    samples = []
    img_x = data[2][0][1]
    data = data[0] #Sacamos la serie que representa la columna de bounding boxes
    if type(data[0][0]) == ultralytics.engine.results.Boxes:
        for d in data:
            xywh = d.xywh #tensor
            for value in xywh:
                samples.append(value[0])
    elif type(data[0][0]) == ultralytics.engine.results.OBB:
        for d in data:
            xywhr = d.xywhr #tensor
            for value in xywhr:
                samples.append(value[0]) 

    #Clustering by KMeans is done to obtain the number of bounding boxes for the x
    x2 = []
    samples_mean = np.mean(samples)
    for s in samples:
        x2.append(abs(s-samples_mean))
    deviation = np.mean(x2)
    print(deviation)
    print(samples_mean)
    if deviation>=img_x/6:
        #Hay 2 peces
        print("Hay 2 truchas")
        return 2,samples_mean
    else:
        print("Hay 1 trucha")
        return 1
    


if __name__ == "__main__":
    import YOLOv8Model
    modelo = YOLOv8Model.Yolo_Model()
    modelo.set_task("detect")
    modelo.video_inference(source="resources/videos/23_NT_R1_J1_P9_10.mp4")
    data = modelo.get_boxes_results()
    estimate_fish_number(data=data)
    pass