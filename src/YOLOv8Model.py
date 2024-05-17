"""
Modulo que define una interfaz util para cargar, predecir y obtener los resultados de un modelo YOLOv8 preentrenado
@author: Diego Aceituno Seoane
@Mail: diego.aceituno@alumnos.upm.es
"""
from ModelVersion import Version
import torch
import openvino as ov
from ultralytics import YOLO
class Yolo_Model:


    def __init__(self,VersionType:str=None,obb=False):
        """
        Configura y carga el modelo
        """
        self.model = None
        core = ov.Core()
        devices1 = core.available_devices
        cuda_bool = torch.cuda.is_available()
        path = "src/models/normal/" #Path donde se encuentran los modelos, TODO ser capaz de buscarlos/descargarlos del repositorio de Diego
        self.task="detect"
        self.last_prediction_results=None
        if obb == True:
            path = "src/models/obb/"
            task="obb"
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


    def video_inference(self,source = None,save = False):
        """
            Método para realizar inferencia sobre un video y obtener resultados
        """
        if source==None:
            raise Exception("No se ha indicado el path del video")
        try:
            resultados = self.model.predict(source=source,save=save,stream=True) #Returns a results generator with stream=True
            for r in resultados:
                continue
            pass
        except:
            raise Exception(f"Problema en la inferencia sobre video:",str(source))
    
    

if __name__ == '__main__':
    modelo = Yolo_Model()
    modelo.video_inference(source="resources/videos/23_NT_R1_J1_P9_10.mp4",save=True)