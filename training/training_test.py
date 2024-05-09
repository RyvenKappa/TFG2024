def main():

    from ultralytics import YOLO
    model = YOLO("yolov8n-obb.pt")
    model.to('cuda')
    model.train(data='training/sp_obb.yaml',epochs=400,imgsz=640,device='0',batch=-1,cache=False,patience=0)

if __name__=="__main__":
    main()
