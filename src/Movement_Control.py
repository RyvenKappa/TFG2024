
import pandas as pd

class Movement_Estimator():
    """
    Clase que se usa para calcular la cantidad de movimientos que han ocurrido según heurística
    """
    
    def __init__(self,data:pd.DataFrame=None) -> None:
        self.data = data
    
    def detect_fish_movements(self):
        """
        Metodo para realizar el calculo en función de los datos
        """
        resultado = []
        for side in self.data.columns:
            #Por logica el primer resultado es el de la izquierda
            resultado.append(self.__diff_calculations(self.data[side]))
    
    def __diff_calculations(self,series:pd.Series)->pd.DataFrame:
        
        side_data = pd.json_normalize(series)
        
        side_data = side_data.diff()
        side_data['area_gradient'] = np.gradient(side_data['area'].copy())
        side_data["centroide_diff"] = np.sqrt(side_data['centroideX']**2 + side_data['centroideY']**2)
        side_data.insert(1,'centroide_diff',side_data.pop('centroide_diff'))
        side_data = side_data.drop(columns="centroideX")
        side_data = side_data.drop(columns="centroideY")
        movimientos = 0
        count = 0
        for i in side_data["area"]:
            if i< -35000 and side_data['blur'][count]<-5 and side_data["centroide_diff"][count]>20:
                movimientos +=1
            count +=1
        return side_data

    def set_data(self,data:pd.DataFrame=None):
        """
        Metodo para cambiar los datos
        """
        if data==None:
            return
        self.data=data

    def get_estimation_list(self)-> list:
        """
        Devuelve una lista con los frames en los que ha sucedido un movimiento
        """

    def get_estimation(self) -> int:
        """
        Devuelve un entero con los frames en los que ha sucedido un movimiento
        """


if __name__ == "__main__":
    import pandas as pd
    from YOLOv8Model import Yolo_Model
    from ResultsProcessor import Data_Processor
    import numpy as np
    modelo = Yolo_Model()
    #modelo.set_task("obb")
    modelo.video_inference(source="resources/videos/23_NT_R1_J1_P5_6.mp4")
    data = modelo.get_boxes_results()
    procesor = Data_Processor()
    resultado = procesor.dataframe_builder(data=data)
    estimador = Movement_Estimator(data=resultado)
    resultado = estimador.detect_fish_movements()
    pass