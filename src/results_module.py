from multiprocessing import Process
import pandas as pd
from Fish_Estimator import estimate_fish_number
import ultralytics

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
                    Procesamos el siguiente fotograma para los datos globales
                """


    def __frame_processing(self,frame_data=None):
        """
        Metodo privado para procesar cada fotograma,obtenemos un diccionario con 1 pareja key-value por pez, con valor diccionario con:

            1. area
            2. posición centroide
            3. Ángulo de rotación
            4. Blurrness, lo sacamos independientemente

        """

        self.proccesed_result = [None,None]
        boxes = frame_data[0]
        orig_img = frame_data[1]
        size = frame_data[2]
        if type(boxes)==ultralytics.engine.results.OBB:
            self.__best_box_obb_detect(boxes,orig_img,size)
        else:
            self.__best_box_data_detect(boxes,orig_img,size)