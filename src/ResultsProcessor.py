"""
Modulo que define una utilidad para procesar los resultados obtenidos de una inferencia con un modelo de YoloV8, ya sea en detection o en obb.
Transforma la información obtenida en campos de las boxes/obb_boxes
@Author: Diego Aceituno Seoane
@Mail: diego.aceituno@alumnos.upm.es
"""
import ultralytics.engine
import ultralytics.engine.results
from YOLOv8Model import Yolo_Model
import pandas as pd
import json
import ultralytics
from Fish_Estimator import estimate_fish_number
import gc

class Data_Processor():

    def dataframe_builder(self,data:pd.DataFrame=None) -> dict:
        """
        Metodo para transformar datos del resultado de inferencia YOLO a DataFrame.
        """
        if data is None or data.empty:
            raise Exception("No se han pasado datos para transformar a JSON")
        
        resultado = {
                        'left':[],
                        'right':[]
                    }
        #Se analiza si hay 1 o 2 peces y si hay 2 peces se proceden a marcar
        self.fish_number,self.mean = estimate_fish_number(data)

        for frame in data.values:
            self.__frame_processing(frame_data=frame)
            resultado['left'].append(self.proccesed_result[0])
            resultado['right'].append(self.proccesed_result[1])
        datos = pd.DataFrame(resultado)
        if self.fish_number == 1:
            datos = datos.drop(columns='right')

        return datos
        

    def __frame_processing(self,frame_data=None) -> dict:
        """
        Metodo privado para procesar cada fotograma,obtenemos un diccionario con 1 pareja key-value por pez, con valor diccionario con:

            1. area
            2. posición centroide
            3. Ángulo de rotación
            4. Blurrness, lo sacamos independientemente

        """

        self.proccesed_result = [None,None]
        boxes = frame_data[0]
        #orig_img = frame_data[1]

        if type(boxes)==ultralytics.engine.results.OBB:
            self.__best_box_obb_detect(boxes)
        else:
            self.__best_box_data_detect(boxes)
        



    def __blurness_estimation(self,image):
        """
        Metodo privado para estimar el blurr que tiene una imagen a través de un filtro laplaciano
        """
        pass

    def __best_box_data_detect(self,boxes):
        """
        Metodo privado para calcular la información de la mejor caja del fotograma para cada pez, independiente del numero de peces
        Metodo diseñado para resultados de ultralytics en modo detect
        """
        if self.fish_number > 1: #TODO hacer el blur por zona de la imagen
            #Diferenciamos entre izquierda y derecha
            self.proccesed_result[0] = dict()
            self.proccesed_result[1] = dict()
            if boxes.xywh.size()[0] == 0:
                return
            left_best = 0
            right_best = 0
            i = 0
            for box in boxes.xywh:
                if box[0]> self.mean:
                    #Estamos a la derecha
                    if boxes.conf[i] > right_best: right_best = i
                else:
                    #Estamos a la izquierda
                    if boxes.conf[i] > left_best: left_best = i
                i += 1
            #Añadimos los datos de la izquierda
            self.proccesed_result[0]["area"] = (boxes.xywh[left_best][2]*boxes.xywh[left_best][3]).item()
            self.proccesed_result[0]["centroideX"] = boxes.xywh[left_best][0].item()
            self.proccesed_result[0]["centroideY"] = boxes.xywh[left_best][1].item()
            self.proccesed_result[0]["angulo"] = 0
            self.proccesed_result[0]["blur"] = 0 #TODO perndiente
            #Añadimos los datos de la derecha
            self.proccesed_result[1]["area"] = (boxes.xywh[right_best][2]*boxes.xywh[right_best][3]).item()
            self.proccesed_result[1]["centroideX"] = boxes.xywh[right_best][0].item()
            self.proccesed_result[1]["centroideY"] = boxes.xywh[right_best][1].item()
            self.proccesed_result[1]["angulo"] = 0
            self.proccesed_result[1]["blur"] = 0 #TODO perndiente
        else:
            #No diferenciamos, solo llenamos izquierda
            self.proccesed_result[0] = dict()
            self.proccesed_result[1] = None
            if boxes.xywh.size()[0] == 0:
                return
            left_best = 0
            i = 0
            for box in boxes.xywh:
                if boxes.conf[i] > left_best: left_best = i
                i += 1
            #Añadimos los datos de la izquierda
            self.proccesed_result[0]["area"] = (boxes.xywh[left_best][2]*boxes.xywh[left_best][3]).item()
            self.proccesed_result[0]["centroideX"] = boxes.xywh[left_best][0].item()
            self.proccesed_result[0]["centroideY"] = boxes.xywh[left_best][1].item()
            self.proccesed_result[0]["angulo"] = 0
            self.proccesed_result[0]["blur"] = 0 #TODO perndiente



    def __best_box_obb_detect(self,boxes):
        """
        Metodo privado para calcular la información de la mejor caja del fotograma para cada pez, independiente del numero de peces
        Metodo diseñado para resultados de ultralytics en modo Oriented Bounding Boxes
        """
        if self.fish_number > 1: #TODO hacer el blur por zona de la imagen
            #Diferenciamos entre izquierda y derecha
            self.proccesed_result[0] = dict()
            self.proccesed_result[1] = dict()
            if boxes.xywhr.size()[0] == 0:
                return
            left_best = 0
            right_best = 0
            i = 0
            for box in boxes.xywhr:
                if box[0]> self.mean:
                    #Estamos a la derecha
                    if boxes.conf[i] > right_best: right_best = i
                else:
                    #Estamos a la izquierda
                    if boxes.conf[i] > left_best: left_best = i
                i += 1
            #Añadimos los datos de la izquierda
            self.proccesed_result[0]["area"] = (boxes.xywhr[left_best][2]*boxes.xywhr[left_best][3]).item()
            self.proccesed_result[0]["centroideX"] = boxes.xywhr[left_best][0].item()
            self.proccesed_result[0]["centroideY"] = boxes.xywhr[left_best][1].item()
            self.proccesed_result[0]["angulo"] = boxes.xywhr[left_best][4].item()
            self.proccesed_result[0]["blur"] = 0 #TODO perndiente
            #Añadimos los datos de la derecha
            self.proccesed_result[1]["area"] = (boxes.xywhr[right_best][2]*boxes.xywhr[right_best][3]).item()
            self.proccesed_result[1]["centroideX"] = boxes.xywhr[right_best][0].item()
            self.proccesed_result[1]["centroideY"] = boxes.xywhr[right_best][1].item()
            self.proccesed_result[1]["angulo"] = boxes.xywhr[right_best][4].item()
            self.proccesed_result[1]["blur"] = 0 #TODO perndiente
        else:
            #No diferenciamos, solo llenamos izquierda
            self.proccesed_result[0] = dict()
            self.proccesed_result[1] = None
            if boxes.xywhr.size()[0] == 0:
                return
            left_best = 0
            i = 0
            for box in boxes.xywhr:
                if boxes.conf[i] > left_best: left_best = i
                i += 1
            #Añadimos los datos de la izquierda
            self.proccesed_result[0]["area"] = (boxes.xywhr[left_best][2]*boxes.xywhr[left_best][3]).item()
            self.proccesed_result[0]["centroideX"] = boxes.xywhr[left_best][0].item()
            self.proccesed_result[0]["centroideY"] = boxes.xywhr[left_best][1].item()
            self.proccesed_result[0]["angulo"] = boxes.xywhr[left_best][4].item()
            self.proccesed_result[0]["blur"] = 0 #TODO perndiente






if __name__ == "__main__":
    modelo = Yolo_Model()
    #modelo.set_task("obb")
    modelo.video_inference(source="resources/videos/23_NT_R1_J1_P9_10.mp4")
    data = modelo.get_boxes_results()
    procesor = Data_Processor()
    resultado = procesor.dataframe_builder(data=data)
    resultado
    pass