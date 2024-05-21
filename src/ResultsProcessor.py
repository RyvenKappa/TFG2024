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

class Data_Processor():

    def json_builder(self,data:pd.DataFrame=None):
        """
        Metodo para transformar datos del resultado de inferencia YOLO a JSON.
        Data es un dataframe
        """
        if data is None or data.empty:
            raise Exception("No se han pasado datos para transformar a JSON")
        resultado = dict()
        n = 0
        for frame in data.values:
            resultado[n] = self.__frame_processing(frame_data=frame)
            n+=1
            pass#TODO continuar para construir el JSON

    def __frame_processing(self,frame_data=None) -> dict:
        """
        Metodo privado para procesar cada fotograma,obtenemos:
            1.area
            2.posición centroide
            3.Ángulo de rotación
            4.Blurrness, lo sacamos independientemente
        """
        proccesed_result = dict()
        if type(frame_data[0])==ultralytics.engine.results.OBB:
            print("soy OBB")

        elif type(frame_data[0])==ultralytics.engine.results.Boxes:
            print("soy de normal")

        #frame_data[0]

        pass

    def __blurness(self,image):
        pass

if __name__ == "__main__":
    modelo = Yolo_Model()
    modelo.set_task("detect")
    modelo.video_inference(source="resources/videos/23_NT_R1_J1_P9_10.mp4")
    data = modelo.get_boxes_results()
    procesor = Data_Processor()
    procesor.json_builder(data=data)
    pass