import numpy as np
"""
    Calorias obtenidas de diferentes comidas en columna, calcular el
    porcentaje de calorias obtenidas por carbohidratos, proteina o grasa
"""
A = np.array([[56.0,0.0,4.4,68.0],
              [1.2,104.0,52.0,8.0],
              [1.8,135.0,99.0,0.9]])
print(A)

calc = A.sum(axis=0)
print(calc)

percentage = 100*A/calc.reshape(1,4)
print(percentage)