from multiprocessing import Process
import pandas as pd
from Fish_Estimator import estimate_fish_number

class Data_Processor(Process):

    def __init__(self,recepcion_endpoint,update_endpoint,frames_number):
        super().__init__()
        self.data_enpoint = recepcion_endpoint
        self.update_enpoint = update_endpoint
        self.datos_entrantes = []
        self.trout_number = None
        self.mean = None
        self.frames_number = frames_number

    def run(self):
        frame = 0
        while True:
            if self.data_enpoint.poll():
                datos = self.data_enpoint.recv()
                self.update_enpoint.send(frame)#Envio el frame por el que voy
                if datos is not None:
                    frame = frame + 1
                    self.datos_entrantes.append(datos)
                if len(self.datos_entrantes)==50 and self.trout_number==None:
                    self.trout_number,self.mean = estimate_fish_number(pd.DataFrame(self.datos_entrantes))
                    self.mean = int(self.mean)
                    print(f"Tenemos {self.trout_number} truchas y una mediana de: {self.mean}")
            elif self.trout_number!=None:
                """
                    Procesamos fotogramas hasta que hemos acabado
                """


