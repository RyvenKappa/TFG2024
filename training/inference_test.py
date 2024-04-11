from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import math

model = YOLO("runs/detect/train/weights/best.pt")
#Con el argumento save podemos decirle que guarde el video o las imágenes según si es solo un fotograma
#Con save_crop le decimos que guarde todos los recordes de todas las bounding box, sin información añadida y sin poder acceder al bucle
results = model.predict(source="resources/videos/test2.mp4")


#11 de Abril: planteamiento: sacar un registro de prueba de las bounding box, su centroide respecto al frame y una gráfica de distribución del centroide

#1 Obtenemos las bounding boxes y calculamos area posición y lo guardamos
#file_output = open("boxes.txt",mode="w")
bounding_box_data = []
centroxList = []
centroyList = []
area = []
for r in results:
    #1 RESULTADO POR FOTOGRAMA
    #file_output.write(str(r.boxes)+"\n\n\n")
    i = 0
    boxes = r.boxes.xywh.tolist() #Formato de salida tipo [centroidex,centroidey,areax,areay]
    confidences = r.boxes.conf.tolist()
    for bb in boxes:
        #X Boxes por
        if confidences[i]>=0.4:
            #[posicion, area]
            centrox = bb[0]
            centroxList.append(centrox)
            centroy = bb[1]
            centroyList.append(centroy)
            area.append(bb[2]*bb[3])
            area2 = [(i*100/(1920*1080)) for i in area]
            data = [centrox,centroy,bb[2]*bb[3]]
            bounding_box_data.append(data) # Podemos hacer esto con un pandas
        i=i+1

file_output = open("SalidaPrueba.txt",mode="w")
file_output.write(str(bounding_box_data))
file_output.close()

#Prueba de gráfica
#Meter el color transformando areas a rgb
plt.scatter(centroxList,centroyList,s=20,c=area2,cmap="gray_r")
plt.title("Centroid and area of bounding boxes for each frame")
plt.xlabel('posición x')
plt.ylabel('posición y')
plt.colorbar()
plt.grid()
plt.xlim(0,1920)
plt.ylim(0,1080)
plt.show()

# Sacar lista para cada lado
distanciasIzquierda = []
distanciasDerecha = []
DcentroidxList = []
DcentroidyList = []
IcentroidxList = []
IcentroidyList = []

#Sacar puntos de cada lado
frames = 0
for r in results:
    boxes = r.boxes.xywh.tolist() #Formato de salida tipo [centroidex,centroidey,areax,areay]
    i = 0
    confidences = r.boxes.conf.tolist()
    for bb in boxes:
        if confidences[i]>=0.4:
            if bb[0]>916:
                #La bounding box esta en la derecha, me guardo su centroide
                DcentroidxList.append(bb[0])
                DcentroidyList.append(bb[1])
            else:
                #la bounding box esta en la izquierda, me guardo su centroide
                IcentroidxList.append(bb[0])
                IcentroidyList.append(bb[1])
        i = i + 1
    frames = frames + 1

DcentroidxListDiff = np.diff(DcentroidxList)
DcentroidyListDiff = np.diff(DcentroidyList)
IcentroidxListDiff = np.diff(IcentroidxList)
IcentroidyListDiff = np.diff(IcentroidyList)
#Les inserto un primer valor 0
DcentroidxListDiff = np.insert(DcentroidxListDiff,0,0)
DcentroidyListDiff = np.insert(DcentroidyListDiff,0,0)
IcentroidxListDiff = np.insert(IcentroidxListDiff,0,0)
IcentroidyListDiff = np.insert(IcentroidyListDiff,0,0)

RightFinal = []
LeftFinal = []
#Calculamos modulo de los cambios
for x, y in zip(DcentroidxListDiff,DcentroidyListDiff):
    distance = math.sqrt((x**2)+(y**2))
    RightFinal.append(distance)
for x, y in zip(IcentroidxListDiff,IcentroidyListDiff):
    distance = math.sqrt((x**2)+(y**2))
    LeftFinal.append(distance)

#Arreglar que el array numpy sea un vector, no un (x,)
plt.plot(RightFinal)
plt.show()
plt.plot(LeftFinal)
plt.show()
