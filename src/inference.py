import multiprocessing as mp
from ultralytics import YOLO
import numpy as np
import pickle

class Video_Inference(mp.Process):

    def __init__(self,modelo:YOLO,pipe_input_endpoint,task,source=None,save=False):
        super().__init__()
        self.model = modelo
        self.endpoint = pipe_input_endpoint
        self.task = task
        self.source = source
        self.save = save


    def run(self):
        """
            Método para realizar inferencia sobre un video y obtener resultados
        """
        mensaje = None
        if self.source==None:
            raise Exception("No se ha indicado el path del video")
        try:
            resultados = self.model.predict(source=self.source,save=self.save,stream=True) #Returns a results generator with stream=True
            if self.task == "obb":
                for r in resultados:
                    mensaje = np.array([r.obb.numpy(),r.orig_img,r.orig_shape],dtype=object)
                    self.endpoint.send(mensaje)
            else:
                for r in resultados:
                    mensaje = np.array([pickle.dumps(r.boxes),r.orig_img,r.orig_shape],dtype=object)
                    self.endpoint.send(mensaje)
        except Exception as e:
            raise Exception(f"Problema en la inferencia sobre video:",str(self.source))