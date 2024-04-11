from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO("C:/Users/Diego/Documents/TFG2024/runs/detect/train/weights/best.pt")
model.to("cuda")
#Con el argumento save podemos decirle que guarde el video o las imágenes según si es solo un fotograma
#Con save_crop le decimos que guarde todos los recordes de todas las bounding box, sin información añadida y sin poder acceder al bucle
results = model.predict(source="C:/Users/Diego/Documents/TFG2024/resources/videos/test2.mp4")


#11 de Abril: planteamiento: sacar un registro de prueba de las bounding box, su centroide respecto al frame y una gráfica de distribución del centroide

#1 Obtenemos las bounding boxes y calculamos area posición y lo guardamos
#file_output = open("boxes.txt",mode="w")
bounding_box_data = []
centroxList = []
centroyList = []
area = []
for r in results:
    #file_output.write(str(r.boxes)+"\n\n\n")
    i = 0
    boxes = r.boxes.xywh.tolist() #Formato de salida tipo [centroidex,centroidey,areax,areay]
    confidences = r.boxes.conf.tolist()
    for bb in boxes:
        if confidences[i]>=0.4:
            #[posicion, area]
            centrox = bb[0]
            centroxList.append(centrox)
            centroy = bb[1]
            centroyList.append(centroy)
            area.append(bb[2]*bb[3])
            data = [centrox,centroy,bb[2]*bb[3]]
            bounding_box_data.append(data) # Podemos hacer esto con un pandas
            i=i+1

file_output = open("SalidaPrueba.txt",mode="w")
file_output.write(str(bounding_box_data))
file_output.close()

#Prueba de gráfica

plt.scatter(centroxList,centroyList,area)
plt.xlabel('posición x')
plt.ylabel('posición y')
plt.show()
    

