def main():

    from ultralytics import YOLO
    model = YOLO("C:/Users/Diego/Documents/Código/TFG2024/resources/models/yolov8n.pt")
    #model.to('cuda')
    model.train(data='C:/Users/Diego/Documents/Código/TFG2024/training/sp.yaml',epochs=100,imgsz=640,device='cpu',batch=-1,cache=False)

if __name__=="__main__":
    main()
