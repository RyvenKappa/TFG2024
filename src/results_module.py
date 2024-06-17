
from collections.abc import Callable
from multiprocessing import Process
from typing import Any, Iterable, Mapping
import pandas as pd

class Data_Processor(Process):

    def __init__(self,recepcion_endpoint):
        super().__init__()
        self.data_enpoint = recepcion_endpoint

    def run(self):
        while True:
            datos = self.data_enpoint.recv()
