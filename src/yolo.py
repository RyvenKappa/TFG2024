"""
Modulo que define una interfaz util para cargar, predecir y obtener los resultados de un modelo YOLOv8 preentrenado
@author: Diego Aceituno Seoane
@Mail: diego.aceituno@alumnos.upm.es
"""
import torch
import openvino as ov
from ultralytics import YOLO
import pandas as pd
from multiprocessing import Process,Queue
from inference import Video_Inference
class model(Process):
    def video_inference(self,source = None,save = False):
        #Quiero que cree dos procesos y obligue a que se comuniquenc
        cola = Queue()
        inference_tool = Video_Inference()

        pass
        
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