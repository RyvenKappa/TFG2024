"""
Modulo que define una interfaz util para cargar, predecir y obtener los resultados de un modelo YOLOv8 preentrenado
@author: Diego Aceituno Seoane
@Mail: diego.aceituno@alumnos.upm.es
"""
import torch
import openvino as ov
from ultralytics import YOLO
from sys import getsizeof
import pandas as pd

class Yolo_Model:


    def __init__(self,VersionType:str=None,obb=False):
        """
        Configura y carga el modelo inicial según la configuración
        """
        self.model = None
        if obb == True:
            self.task = "obb"
        else:
            self.task="detect"
        self.last_prediction_results=None
        self.__set_model()
        


    def video_inference(self,source = None,save = False):
        """
            Método para realizar inferencia sobre un video y obtener resultados
        """
        if source==None:
            raise Exception("No se ha indicado el path del video")
        try:
            resultados = self.model.predict(source=source,save=save,stream=True) #Returns a results generator with stream=True
            data = []
            if self.task == "obb":
                for r in resultados:
                    data.append([r.obb,r.orig_img,r.orig_shape])
            else:
                for r in resultados:
                    data.append([r.boxes,r.orig_img,r.orig_shape])
            self.last_prediction_results = pd.DataFrame(data)
            #print(getsizeof(self.last_prediction_results))
        except Exception as e:
            raise Exception(f"Problema en la inferencia sobre video:",str(source))
        
    def get_boxes_results(self):
        return self.last_prediction_results
    
    def set_task(self,task):
        """
        Metodo para configurar el tipo de trabajo del modelo, si no ha cambiado, no se realiza el cambio
        """
        if task not in ["detect","obb"]:
            raise Exception(f"El tipo de trabajo ({task}) que se esta pidiendo no es el correcto.")
        else:
            if task != self.task:
                print(f"Se va a cambiar a un modo de trabajo {task}")
                self.task = task
                self.__set_model()
            else:
                print("Ya esta activo este tipo de trabajo")

    def __set_model(self):
        """
            Metodo privado de la clase que se usa para configurar el modelo según hardware y según el tipo de trabajo marcado en el objeto
        """
        core = ov.Core()
        devices1 = core.available_devices
        cuda_bool = torch.cuda.is_available()
        path = "src/models/normal/" #Path donde se encuentran los modelos, TODO ser capaz de buscarlos/descargarlos del repositorio de Diego
        
        if self.task == "obb":
            path = "src/models/obb/"
        try:
            if cuda_bool:
                print(f"Modelo a cargar de pt,gpu=",torch.get_device(0))
                self.model = YOLO(f"{path}best.pt",task=self.task)
                self.model.to('cuda')
            else:
                if len(devices1)>=2:
                    """
                    CPU y una GPU de intel, y otra de amd o desconocida
                    """
                    print(f"Modelo a cargar de openvino, gpu=",{core.get_property("GPU","FULL_DEVICE_NAME")})
                    self.model = YOLO(f"{path}best_openvino_model",task=self.task)
                else:
                    """
                    CPU solo, modelo ONNX
                    """
                    print(f"Modelo a cargar de ONNX, cpu=",{core.get_property("CPU","FULL_DEVICE_NAME")})
                    self.model = YOLO(f"{path}best.onnx",task=self.task)
        except:
            print("Problema cargando un modelo eficiente, cargando modelo por defecto tipo PyTorch")
            self.model = YOLO(f"{path}best.pt",task=self.task)

        
    

if __name__ == '__main__':
    modelo = Yolo_Model(obb=False)
    modelo.set_task("detect")
    #modelo.set_task("obb")
    modelo.video_inference(source="resources/videos/23_NT_R1_J1_P9_10.mp4")
    data = modelo.get_boxes_results()
    data.to_excel("cosa.xlsx")
    modelo.set_task("detect")
    modelo.video_inference(source="resources/videos/23_NT_R1_J1_P9_10.mp4")
    data = modelo.get_boxes_results()