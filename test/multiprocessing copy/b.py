from multiprocessing import Process
from multiprocessing.connection import Connection
import multiprocessing as mp

class Process_B(Process):

    def __init__(self,connection):
        super().__init__()
        self.endPoint = connection
    
    
    def run(self):
        while True:
            informacion = self.endPoint.recv()
            print(f"{informacion}")