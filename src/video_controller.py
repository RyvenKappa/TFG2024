import multiprocessing as mp
import cv2 as cv
import time
import numpy as np

class Controller(mp.Process):
    def __init__(self, control_output_endpoint, frame_input_enpoint,video_path) -> None:
        super().__init__()
        self.control_output_endpoint = control_output_endpoint
        self.frame_input_endpoint = frame_input_enpoint
        self.frame = 0
        self.play = False
        self.video_path = video_path
        

    def run(self) -> None:
        """
            Método de bucle principal
        """
        self.cap = cv.VideoCapture(self.video_path)
        self.fps = self.cap.get(cv.CAP_PROP_FPS)
        print(self.fps)
        self.t = 1/(self.fps*0.8)
        while True:
            if self.control_output_endpoint.poll():
                self.procesar_mensaje(self.control_output_endpoint.recv())
            else:
                if self.play:
                    ret, self.frame = self.cap.read()
                    if ret:
                        self.frame_input_endpoint.send(self.next_frame())
                        time.sleep(self.t)
                    else:
                        self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                else:
                    self.procesar_mensaje(self.control_output_endpoint.recv())
    
    def procesar_mensaje(self,mensaje):
        """
            Método para cambiar funcionamiento según mensaje de control
        """
        if mensaje==True:
            self.play = True
        elif mensaje==False:
            self.play = False
        elif type(mensaje) is int:
            self.cap.set(cv.CAP_PROP_POS_FRAMES, mensaje)
            self.frame_input_endpoint.send(self.next_frame())

    def next_frame(self):
            """
                Realiza las operaciones para que el frame que se pasa se pueda aplicar directamente a la textura sin necesidad de procesado de la interfaz
            """
            frame_rgb = cv.resize(self.frame,(720,480))
            frame_rgb = cv.cvtColor(frame_rgb,cv.COLOR_BGR2RGB)
            data = frame_rgb.flatten().astype(np.float32)/255
            return data
