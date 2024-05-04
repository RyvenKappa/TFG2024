from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.patches as mpatches
model = YOLO("runs/detect/trainOriginal/weights/best_openvino_model/")

#model = YOLO("runs/detect/train/weights/best.onnx")
#Con el argumento save podemos decirle que guarde el video o las imágenes según si es solo un fotograma
#Con save_crop le decimos que guarde todos los recordes de todas las bounding box, sin información añadida y sin poder acceder al bucle
results = model.predict(source="resources/videos/23_NT_R1_J1_P9_10.mp4",save=True)


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
        if confidences[i]>=0.80:
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
#scatter = plt.scatter(centroxList,centroyList,s=20,c=area2,cmap="viridis")
plt.figure(figsize=(16,9))
scatter = plt.scatter(centroxList,centroyList,s=20)
#plt.title("Centroid and area of each Bounding Box")
#plt.xlabel('X Position in Frame [0,1920]')
#plt.ylabel('Y Position in Frame [0,1080]')
#cbar = plt.colorbar()
#cbar.ax.set_ylabel('Frame area percentage occupied by the trout')
plt.grid()
plt.xlim(0,1920)
plt.ylim(0,1080)
plt.gca().invert_yaxis() #Invierto el axis y ya que la la vertical empieza por 0
#plt.show()

# Sacar lista para cada lado
distanciasIzquierda = []
distanciasDerecha = []
DcentroidxList = []
DcentroidyList = []
IcentroidxList = []
IcentroidyList = []
IareaList = []
DareaList = []
#Sacar puntos de cada lado
frames = 0
for r in results:
    boxes = r.boxes.xywh.tolist() #Formato de salida tipo [centroidex,centroidey,areax,areay]
    i = 0
    confidences = r.boxes.conf.tolist()
    for bb in boxes:
        if confidences[i]>=0.80:
            if bb[0]>916:
                #La bounding box esta en la derecha, me guardo su centroide
                DcentroidxList.append(bb[0])
                DcentroidyList.append(bb[1])
                DareaList.append((bb[2]*bb[3]*100)/(1920*1080))
            else:
                #la bounding box esta en la izquierda, me guardo su centroide
                IcentroidxList.append(bb[0])
                IcentroidyList.append(bb[1])
                IareaList.append((bb[2]*bb[3]*100)/(1920*1080))
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



"""
    En este segmento, vamos a plottear 2 cosas:
    1- La caantidad de movimiento que se ha desplazado la bounding box detectada de la trucha en ese fotograma
    2- La area de la  bounding box detectada
    Para poder trabajar con fotogramas, hay que hacer que cuando no detectemos a la trucha, asumir que sigue ahí y repetir la misma bounding box
"""
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(RightFinal,color="tab:blue",label="Distance")
ax2.plot(DareaList,'--',color="tab:red",label="Area")
plt.title("Distance changes and area changes of the Right Trout")
ax1.set_xlabel('Frame number')
ax1.set_ylabel('Distance moved by the fish')
ax2.set_ylabel('Area of the Bounding Box')
#Añadimos los artist para la leyenda
blue_distance = mpatches.Patch(color='blue',label='Distance')
red_area = mpatches.Patch(color='red',label='Area')
plt.legend(handles=[blue_distance,red_area])
plt.xlim(0,len(RightFinal)-1)
plt.show()
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(LeftFinal,color="tab:blue",label="Distance")
ax2.plot(IareaList,'--',color="tab:red",label="Area")
plt.title("Distance changes and area changes of the Left Trout")
ax1.set_xlabel('Frame number')
ax1.set_ylabel('Distance moved by the fish')
ax2.set_ylabel('Area of the Bounding Box')
plt.legend(['Distance','Area'])
#Añadimos los artist para la leyenda
blue_distance = mpatches.Patch(color='blue',label='Distance')
red_area = mpatches.Patch(color='red',label='Area')
plt.legend(handles=[blue_distance,red_area])
plt.xlim(0,len(LeftFinal)-1)
plt.show()


# plt.figure(figsize=(10,5))
# plt.plot(LeftFinal,color="tab:blue",label="Distancia")
# plt.xlabel('# fotograma')
# plt.ylabel('Desplazamiento en píxeles')
# plt.xlim(0,len(LeftFinal)-1)
# plt.ylim(0,350)
# #plt.show()

# plt.figure(figsize=(10,5))
# plt.plot(IareaList,color="tab:blue",label="Area")
# plt.xlabel('# fotograma')
# plt.ylabel('Área en % del tamaño de la imagen')
# plt.xlim(0,len(LeftFinal)-1)
# plt.ylim(0,20)
# #plt.show()

# plt.figure(figsize=(10,5))
# plt.plot(RightFinal,color="tab:blue",label="Distancia")
# plt.xlabel('# fotograma')
# plt.ylabel('Desplazamiento en píxeles')
# plt.xlim(0,len(RightFinal)-1)
# plt.ylim(0,350)
# #plt.show()

# plt.figure(figsize=(10,5))
# plt.plot(DareaList,color="tab:blue",label="Area")
# plt.xlabel('# fotograma')
# plt.ylabel('Área en % del tamaño de la imagen')
# plt.xlim(0,len(RightFinal)-1)
# plt.ylim(0,20)
# plt.show()
