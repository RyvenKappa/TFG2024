def main():

    from ultralytics import YOLO
    model = YOLO("yolov8n-obb.pt")
    #model.to('cuda')
    model.train(data='C:/Users/Diego/Documents/Codigo/TFG2024/training/sp.yaml',epochs=30,imgsz=640,device='cpu',batch=-1,cache=False)

if __name__=="__main__":
    main()
