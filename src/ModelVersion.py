"""
Modulo usado para definir los diferentes tipos de deploys para el modelo
preentrenado teniendo en cuenta el Hardware disponible
@Author: Diego Aceituno Seoane
@Mail: diego.aceituno@alumnos.upm.es
"""
from enum import Enum

class Version(Enum):
    CUDA = 'CUDA'
    ONNX = 'ONNX'
    OPENVINO = 'OPENVINO'


if __name__ == '__main__':
    print(Version.CUDA.name)
    