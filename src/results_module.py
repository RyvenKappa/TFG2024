
from collections.abc import Callable
from multiprocessing import Process
from typing import Any, Iterable, Mapping
import pandas as pd

class Data_Processor(Process):

    def __init__(self):
        super().__init__()