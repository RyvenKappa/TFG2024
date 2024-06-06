from multiprocessing import Process
from multiprocessing.connection import Connection
import multiprocessing as mp

class Process_B(Process):

    def __init__(self,connection:mp.Queue):
        super().__init__()
        self.endPoint = connection
    
    
    def run(self):
        while True:
            informacion = self.endPoint.get()
            print(f"{informacion}")