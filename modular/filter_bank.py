import cv2
import numpy as np

class filters:
    def proccess_frame(self,frame):
        """
            procesa un frame y lo devuelve
        """
        gray  = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),0)
        ret, thresh = cv2.threshold(gray,50,255,0)
        thresh = cv2.bitwise_not(thresh)
        c,hierarchy = cv2.findContours(thresh,1,2)
        contours = sorted(c,key = cv2.contourArea,reverse=True)[:10]
        rectangles = []
        for cnt in contours:
                area = cv2.contourArea(cnt)
                if area>100000:
                    x,y,w,h = cv2.boundingRect(cnt)
                    cnt_len = cv2.arcLength(cnt,True)
                    contorno = cv2.approxPolyDP(cnt,0.06*cnt_len,closed=True)
                    ratio = float(w)/h
                    if ratio>=0.6 and ratio <= 1.5:
                        rect = cv2.minAreaRect(cnt)
                        box = cv2.boxPoints(rect)
                        box = np.intp(box)
                        rectangles.append(box)
        cv2.drawContours(frame,rectangles,-1,(0,255,0),3)
    
    def resolve_contours(self,frame):
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(gray,127,255,0)
        contornos,h = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame,contornos,-1,(0,255,0),3)
        return frame