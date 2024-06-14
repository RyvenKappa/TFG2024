import multiprocessing as mp
import pandas as pd
from ultralytics import YOLO

class Video_Inference(mp.Process):

    def __init__(self,modelo:YOLO,queue:mp.Queue,task):
        self.model = modelo
        self.queue = queue
        self.task = task
        super().__init__()

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
                    self.queue.put([r.obb,r.orig_img,r.orig_shape])
            else:
                for r in resultados:
                    self.queue.put([r.boxes,r.orig_img,r.orig_shape])
        except Exception as e:
            raise Exception(f"Problema en la inferencia sobre video:",str(source))
        
if __name__ == '__main__':
    pass