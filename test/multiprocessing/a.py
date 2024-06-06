from multiprocessing import Process
from multiprocessing.connection import Connection
import multiprocessing as mp
import random
import time

class Process_A(Process):
    """
    Este proceso publica valores en un pipe a otro hilo
    """
    def __init__(self,connection:mp.Queue):
        super().__init__()
        self.endPoint = connection

    def run(self):
        while True:
            number = random.randint(1,10)
            self.endPoint.put(number)
            time.sleep(4)