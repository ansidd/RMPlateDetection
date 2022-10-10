import torch
import cv2


def main(path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', './best.pt')
    
    vid = cv2.VideoCapture(path)
    
    ret, frame = vid.read()
    
    while(ret):
        res = model(frame)
        
        for i in res.pred[0]:
            i = i.detach().cpu().numpy()
            x1, y1, x2, y2, conf, c = i
            if conf<0.6:
                continue

            img = cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (255,0,0), 2)
       
        cv2.imshow('video', frame)
        
        if cv2.waitKey(1)==ord('q'):
            break
            
        ret, frame = vid.read()
        
    vid.release()
    
if __name__=="__main__":
    main("./vid.mp4")