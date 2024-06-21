from multiprocessing import Process
import multiprocessing
import pandas as pd
from Fish_Estimator import estimate_fish_number
import ultralytics
import cv2 as cv
import pickle
from Movement_Control import Movement_Estimator
import numpy as np

class Data_Processor(Process):

    def __init__(self,recepcion_endpoint,update_endpoint,frames_number):
        super().__init__()
        self.data_enpoint = recepcion_endpoint
        self.update_enpoint = update_endpoint
        self.datos_entrantes = []
        self.fish_number = None
        self.mean = None
        self.frames_number = frames_number

    def run(self):
        frame = 0
        resultado = {
                        'left':[],
                        'right':[]
                    }
        while True:
            if self.data_enpoint.poll():
                datos = self.data_enpoint.recv()
                self.update_enpoint.send(frame)#Envio el frame por el que voy, mando un int
                if datos is not None:
                    frame = frame + 1
                    self.datos_entrantes.append([pickle.loads(datos[0]),datos[1],datos[2]])
                if len(self.datos_entrantes)==50 and self.fish_number==None:
                    self.fish_number,self.mean = estimate_fish_number(pd.DataFrame(self.datos_entrantes))
                    self.mean = int(self.mean)
                    #print(f"Tenemos {self.fish_number} truchas y una mediana de: {self.mean}")
            elif self.fish_number!=None:
                pass
                """
                    Procesamos el siguiente fotograma para los datos globales
                """
                if len(resultado['left'])!=self.frames_number:
                    try:
                        self.__frame_processing(frame_data=self.datos_entrantes.pop(0))
                        resultado['left'].append(self.proccesed_result[0])
                        resultado['right'].append(self.proccesed_result[1])
                    except:
                        pass
                else:
                    break
        print(f"Frames procesados, con datos:\n{len(resultado['left'])}")
        datos = pd.DataFrame(resultado)
        if self.fish_number == 1:
            datos = datos.drop(columns='right')
        estimador = Movement_Estimator(data=datos.copy())#Estimamos los movimientos
        #Convertimos el dataframe principal en 2 para el proceso principal, 1 por cada lado
        left = pd.json_normalize(datos['left'])
        left = self.centroid_abs(left)
        if self.fish_number == 2:
            right = pd.json_normalize(datos['right'])
            right = self.centroid_abs(right)
        self.update_enpoint.send((estimador.detect_fish_movements(),left,right))#Enviamos los datos procesados al proceso principal


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

    def __blurness_estimation(self,image):
        """
        Metodo privado para estimar el blurr que tiene una imagen a través de un filtro laplaciano
        """
        ddepth = cv.CV_16S
        kernel_size = 3
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        estimation = cv.Laplacian(gray,ddepth,ksize=kernel_size).var()
        return estimation

    def __best_box_data_detect(self,boxes,orig_img,size):
        """
        Metodo privado para calcular la información de la mejor caja del fotograma para cada pez, independiente del numero de peces
        Metodo diseñado para resultados de ultralytics en modo detect
        """
        if self.fish_number > 1:
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
            self.proccesed_result[0]["area"] = ((boxes.xywh[left_best][2]*boxes.xywh[left_best][3]).item())/(size[0]*size[1])*100
            self.proccesed_result[0]["centroideX"] = boxes.xywh[left_best][0].item()
            self.proccesed_result[0]["centroideY"] = boxes.xywh[left_best][1].item()
            self.proccesed_result[0]["width_height_relation"] = (boxes.xywh[left_best][2].item()/boxes.xywh[left_best][3].item())
            self.proccesed_result[0]["angulo"] = 0
            self.proccesed_result[0]["blur"] = self.__blurness_estimation(orig_img[:,0:self.mean])#TODO
            #Añadimos los datos de la derecha
            self.proccesed_result[1]["area"] = ((boxes.xywh[right_best][2]*boxes.xywh[right_best][3]).item())/(size[0]*size[1])*100
            self.proccesed_result[1]["centroideX"] = boxes.xywh[right_best][0].item()
            self.proccesed_result[1]["centroideY"] = boxes.xywh[right_best][1].item()
            self.proccesed_result[1]["width_height_relation"] = (boxes.xywh[right_best][2].item()/boxes.xywh[right_best][3].item())
            self.proccesed_result[1]["angulo"] = 0
            self.proccesed_result[1]["blur"] = self.__blurness_estimation(orig_img[:,self.mean:])
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
            self.proccesed_result[0]["area"] = ((boxes.xywh[left_best][2]*boxes.xywh[left_best][3]).item())/(size[0]*size[1])*100
            self.proccesed_result[0]["centroideX"] = boxes.xywh[left_best][0].item()
            self.proccesed_result[0]["centroideY"] = boxes.xywh[left_best][1].item()
            self.proccesed_result[0]["width_height_relation"] = (boxes.xywh[left_best][2].item()/boxes.xywh[left_best][3].item())
            self.proccesed_result[0]["angulo"] = 0
            self.proccesed_result[0]["blur"] = self.__blurness_estimation(orig_img)



    def __best_box_obb_detect(self,boxes,orig_img,size):
        """
        Metodo privado para calcular la información de la mejor caja del fotograma para cada pez, independiente del numero de peces
        Metodo diseñado para resultados de ultralytics en modo Oriented Bounding Boxes
        """
        if self.fish_number > 1:
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
            self.proccesed_result[0]["area"] = ((boxes.xywhr[left_best][2]*boxes.xywhr[left_best][3]).item())/(size[0]*size[1])*100
            self.proccesed_result[0]["centroideX"] = boxes.xywhr[left_best][0].item()
            self.proccesed_result[0]["centroideY"] = boxes.xywhr[left_best][1].item()
            self.proccesed_result[0]["width_height_relation"] = (boxes.xywhr[left_best][2].item()/boxes.xywhr[left_best][3].item())
            self.proccesed_result[0]["angulo"] = boxes.xywhr[left_best][4].item()
            self.proccesed_result[0]["blur"] = self.__blurness_estimation(orig_img[:,0:self.mean])
            #Añadimos los datos de la derecha
            self.proccesed_result[1]["area"] = ((boxes.xywhr[right_best][2]*boxes.xywhr[right_best][3]).item())/(size[0]*size[1])*100
            self.proccesed_result[1]["centroideX"] = boxes.xywhr[right_best][0].item()
            self.proccesed_result[1]["centroideY"] = boxes.xywhr[right_best][1].item()
            self.proccesed_result[1]["width_height_relation"] = (boxes.xywhr[right_best][2].item()/boxes.xywhr[right_best][3].item())
            self.proccesed_result[1]["angulo"] = boxes.xywhr[right_best][4].item()
            self.proccesed_result[1]["blur"] = self.__blurness_estimation(orig_img[:,self.mean:0])
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
            self.proccesed_result[0]["area"] = ((boxes.xywhr[left_best][2]*boxes.xywhr[left_best][3]).item())/(size[0]*size[1])*100
            self.proccesed_result[0]["centroideX"] = boxes.xywhr[left_best][0].item()
            self.proccesed_result[0]["centroideY"] = boxes.xywhr[left_best][1].item()
            self.proccesed_result[0]["width_height_relation"] = (boxes.xywhr[left_best][2].item()/boxes.xywhr[left_best][3].item())
            self.proccesed_result[0]["angulo"] = boxes.xywhr[left_best][4].item()
            self.proccesed_result[0]["blur"] = self.__blurness_estimation(orig_img)
    
    def centroid_abs(self,side_data:pd.DataFrame)-> pd.DataFrame:
        """
        Método para cambiar las columnas centroidex y centroidey por una sola columna que tenga el modulo de cambio diferencial del centroide
        """
        side_data['centroideX'] = side_data['centroideX'].copy().diff()
        side_data['centroideY'] = side_data['centroideY'].copy().diff()
        side_data["centroide_change"] = np.sqrt(side_data['centroideX']**2 + side_data['centroideY']**2)
        side_data.insert(1,'centroide_change',side_data.pop('centroide_change'))
        side_data = side_data.drop(columns="centroideX")
        side_data = side_data.drop(columns="centroideY")
        side_data['centroide_change'].fillna(0, inplace=True)
        return side_data