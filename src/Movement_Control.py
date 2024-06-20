import numpy as np
import pandas as pd

class Movement_Estimator():
    """
    Clase que se usa para calcular la cantidad de movimientos que han ocurrido según heurística
    """
    
    def __init__(self,data:pd.DataFrame=None) -> None:
        self.data = data
    
    def detect_fish_movements(self):
        """
        Metodo para realizar el calculo en función de los datos, devuelve una lista con 2 o 1 elemento según el
        número de peces que hay
        """
        resultado = []
        for side in self.data.columns:
            #Por logica el primer resultado es el de la izquierda
            resultado.append(self.__diff_calculations(self.data[side]))
        return resultado
    
    def __diff_calculations(self,series:pd.Series)->pd.DataFrame:
        """
            Realiza calculos de movimientos por cada lado del pez
        """
        side_data = pd.json_normalize(series)
        
        #side_data = side_data.diff()
        side_data['frame'] = range(0,len(side_data))
        side_data['area_gradient'] = np.gradient(side_data['area'].copy())
        side_data['centroideX'] = side_data['centroideX'].copy().diff()
        side_data['centroideY'] = side_data['centroideY'].copy().diff()
        side_data["centroide_change"] = np.sqrt(side_data['centroideX']**2 + side_data['centroideY']**2)
        side_data.insert(1,'centroide_change',side_data.pop('centroide_change'))
        side_data = side_data.drop(columns="centroideX")
        side_data = side_data.drop(columns="centroideY")
        pass
        # side_data["centroide_diff"] = np.sqrt(side_data['centroideX']**2 + side_data['centroideY']**2)
        # side_data.insert(1,'centroide_diff',side_data.pop('centroide_diff'))
        # side_data = side_data.drop(columns="centroideX")
        # side_data = side_data.drop(columns="centroideY")
        frame = 0
        movement_data = []
        extra_counter= 0
        self.median = side_data['area'].median()
        contando = False
        extra_array = []
        numeros_array = 0
        original = 0
        for area in side_data['area']:
            if side_data['area_gradient'][frame]<0:
                if not contando:
                    original = frame
                extra_counter = extra_counter + 1
                contando = True
                extra_array.append(frame)
            elif side_data['area_gradient'][frame] >= 0 and contando==True:
                extra_counter = 0
                contando = False
                if 9>len(extra_array)>1:
                    movement_data.append(side_data.iloc[original: frame])
                    numeros_array +=1
                extra_array.clear()
                gradiente_total = 0
            frame = frame + 1
        result_data = []
        for i in movement_data:
            a = self.apply_rules(i)#TODO paralelo
            if a > -1: result_data.append(a)
        return result_data

    def apply_rules(self,data:pd.DataFrame):
        """
        Método que obtiene la lista con las secciones con posible movimiento y decide si hay o no movimiento y lo añade
        1. Checkear si es muy rectangular, entonces denegar
        2. Checkear el cambio total de gradiente
        3. 
        """
        gradiente_total = sum(data['area_gradient']) #No hay el suficiente cambio
        if gradiente_total>-1:
            return -1
        if all(data['area_gradient'] > -0.8):
            return -1
        for rect in data['width_height_relation']: #Muy rectangular
            if rect<0.3 or rect>2.7:
                return -1
        if data.sort_values(by='area',ascending=True)['area'].values[0] > self.median-self.median*0.1:#Me quito situaciones en las que el area no sea lo suficientemente pequeña
            return -1
        
        p_array = []
        
        return data.sort_values(by='blur',ascending=True)['frame'].values[0]
        
        

    def set_data(self,data:pd.DataFrame=None):
        """
        Metodo para cambiar los datos
        """
        if data==None:
            return
        self.data=data