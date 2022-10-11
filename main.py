import torch
import cv2
import numpy as np



obj_count=0

def e_dist(a,b):
    dist = 0
    for i in range(2):
        dist += (a[i]-b[i])**2
    dist = dist**0.5
    return dist


def track_objs(curr_positions, history):
    global obj_count
    
    if len(history)==0:
        n_objects = len(curr_positions)
        obj_count+=1
        object_ids = list(range(obj_count, obj_count+n_objects))
        obj_count+= n_objects
        history = list(zip(object_ids, curr_positions))
        return object_ids, history
    else:
        object_ids = []
        print(curr_positions)
        curr_pos_np = np.array(curr_positions)
        history_pos_np = np.array([obj[1] for obj in history])
        dist = ((curr_pos_np[:,None] - history_pos_np[None,:])**2).sum(axis=2)**0.5
        len_curr_positions = len(curr_pos_np)
        len_history_pos = len(history_pos_np)
        dist_indices_sorted = np.dstack(np.unravel_index(np.argsort((dist).ravel()), (len_curr_positions, len_history_pos)))[0]
        
        object_ids = [-1]*len(curr_positions)
        print(dist)
        print(dist_indices_sorted)
        for i in dist_indices_sorted:
            x, y = i
            if y not in object_ids and dist[x,y]<100:
                if object_ids[x]==-1:
                    object_ids[x] = history[y][0]
                
        for i in range(len(object_ids)):
            if object_ids[i]==-1:
                obj_count+=1
                object_ids[i] = obj_count
                
#         for obj_pos in curr_positions:
#             xi,yi,wi,hi = obj_pos
#             min_dist = 1000000
#             min_id = -1
#             for i,obj in enumerate(history):
#                 xj,yj,wj,hj = obj[1]
#                 if e_dist((xi,yi,wi,hi), (xj,yj,wj,hj))<100:
#                     if e_dist((xi,yi,wi,hi), (xj,yj,wj,hj))<min_dist:
#                         min_dist = e_dist((xi,yi,wi,hi), (xj,yj,wj,hj))
#                         min_id = obj[0]
                
                    
#             if min_id!=-1:
#                 object_ids.append(min_id)
#             else:
#                 obj_count+=1
#                 object_ids.append(obj_count)

        history = list(zip(object_ids, curr_positions))
        return object_ids, history
        

        
def main(path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', './best.pt')
    
    vid = cv2.VideoCapture(path)
    
    ret, frame = vid.read()
    
    tracker_history = {}
    
    while(ret):
        res = model(frame)
        
        curr_positions = []
        
        for i in res.pred[0]:
            i = i.detach().cpu().numpy()
            x1, y1, x2, y2, conf, c = i
            if conf<0.5:
                continue
                
            curr_positions.append([int(x1), int(y1), int(x2-x1), int(y2-y1)])

        if(len(curr_positions)!=0):  
            object_ids, tracker_history = track_objs(curr_positions, tracker_history)
            print(object_ids, tracker_history)
            
        for i in range(len(curr_positions)):               
            x1,y1,w,h = curr_positions[i]
            cv2.rectangle(frame, (int(x1),int(y1-15)), (int(x1)+30,int(y1)-30), (255,255,255), -1)
            cv2.putText(frame,  str(object_ids[i]), (x1, y1-15), cv2.FONT_HERSHEY_PLAIN,1, (255,0,0))
            frame = cv2.rectangle(frame, (int(x1),int(y1)), (int(x1+w),int(y1+h)), (0,255,0), 1)
       
        cv2.imshow('video', frame)
        
        if cv2.waitKey(1)==ord('q'):
            break
            
        ret, frame = vid.read()
        
    vid.release()
    
if __name__=="__main__":
    main("./vid.mp4")