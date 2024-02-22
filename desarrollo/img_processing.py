import cv2
import numpy as np

def process_img(img):
    """
        Procesa una imagen para que sea tratada en el seguimiento de las redes
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #gray = cv2.medianBlur(gray,5)
    gray = cv2.GaussianBlur(img,(5,5),0)
    return gray


if __name__ == "__main__":
    video_stream = cv2.VideoCapture("prueba.MTS")
    ok,img = video_stream.read()

    while ok:
        img = process_img(img)
        cv2.imshow("Processed",img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        ok,img = video_stream.read()