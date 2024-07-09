"""
Modulo que define una interfaz util para cargar, predecir y obtener los resultados de un modelo YOLOv8 preentrenado
@author: Diego Aceituno Seoane
@Mail: diego.aceituno@alumnos.upm.es
"""
import torch
import openvino as ov
from ultralytics import YOLO
from inference import Video_Inference
class model:
    def __init__(self,obb=False) -> None:
        self.model = None
        if obb == True:
            self.task = "obb"
        else:
            self.task="detect"
        self.last_prediction_results=None
        self.cuda = False
        self.__set_model()

    def video_inference(self,pipe_input_endpoint,source = None,save = False):
        #Le dice que empieze a inferir con la tubería de inferencia en la que meter los datos para el de resultados
        self.inference_process = Video_Inference(self.model,pipe_input_endpoint,self.task,source,cuda=self.cuda)
        self.inference_process.start()
    
    def stop_video_inference(self):
        try:
            self.inference_process.terminate()
            self.inference_process.close()
        except:
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

    def get_task(self):
        """
            Getter de la task del objeto yolo
        """
        return self.task

    def __set_model(self):
        """
            Metodo privado de la clase que se usa para configurar el modelo según hardware y según el tipo de trabajo marcado en el objeto
        """
        cargado = False
        core = ov.Core()
        devices = core.available_devices
        print(f"Los dispositivos encontrados son: {devices}")
        try:
            cuda_bool = torch.cuda.is_available()
            if cuda_bool:
                capability = torch.cuda.get_device_capability()
                capability = capability[0] + capability[1]/10
        except:
            cuda_bool = False
        print(f"Boleano de cuda: {cuda_bool}")
        path = "src/models/normal/" #Path donde se encuentran los modelos, TODO ser capaz de buscarlos/descargarlos del repositorio de Diego
        
        if self.task == "obb":
            path = "src/models/obb/"
        try:
            if cuda_bool and capability>=3.7:
                print(f"Modelo a cargar de pt,gpu=",torch.cuda.get_device_name(torch.cuda.current_device()))
                self.model = YOLO(f"{path}best.pt",task=self.task)
                cargado = True
                self.cuda = True
            else:
                self.cuda = False
                if not cuda_bool or not cargado:
                        gpu:str = core.get_property("GPU","FULL_DEVICE_NAME")
                        cpu:str = core.get_property("CPU","FULL_DEVICE_NAME")
                        print(cpu)
                        print(gpu)
                        gpu = gpu.lower()
                        cpu = cpu.lower()
                        if gpu.find("intel")>=0 or cpu.find("intel")>=0 or gpu.find("UHD")>=0:
                            """
                            CPU y una GPU de intel, y otra de amd o desconocida
                            """
                            print(f"Modelo a cargar de openvino, gpu=",{core.get_property("GPU","FULL_DEVICE_NAME")})
                            self.model = YOLO(f"{path}best_openvino_model",task=self.task)
                        else:
                            """
                            CPU solo, modelo ONNX
                            """
                            print("Modelo a cargar de ONNX")
                            print(f"Modelo a cargar de ONNX, cpu=",{core.get_property("CPU","FULL_DEVICE_NAME")})
                            self.model = YOLO(f"{path}best.onnx",task=self.task)
        except Exception as e:
            self.cuda =False
            print(e)
            print("Problema cargando un modelo eficiente, cargando modelo por defecto tipo PyTorch")
            self.model = YOLO(f"{path}best.pt",task=self.task)
