import torch
import cv2
import numpy as np


class Tracker():
    def __init__(self):
        self.obj_count = 0
    #Function to track objects given the current positions and history containing objects and their positions in the previous timestep
    def track_objs(self, curr_positions, history):

        #When there were no objects detected in the previous time step
        if len(history)==0:
            n_objects = len(curr_positions)
            self.obj_count+=1
            object_ids = list(range(self.obj_count, self.obj_count+n_objects))
            self.obj_count+= n_objects
            history = list(zip(object_ids, curr_positions))
            return object_ids, history

        #Idenitfying objects based on current positions
        else:
            object_ids = []
            curr_pos_np = np.array(curr_positions)
            history_pos_np = np.array([obj[1] for obj in history])
            dist = ((curr_pos_np[:,None] - history_pos_np[None,:])**2).sum(axis=2)**0.5
            len_curr_positions = len(curr_pos_np)
            len_history_pos = len(history_pos_np)

            #calculating distance between objects detected in current time step and previous timestep
            dist_indices_sorted = np.dstack(np.unravel_index(np.argsort((dist).ravel()), (len_curr_positions, len_history_pos)))[0]


            #Assigning identifying objects based on difference between current position and previous positions. If an object A's current position is closest to an object B's position the previous step A and B are decided to be the same and A would receive the object ID of B
            object_ids = [-1]*len(curr_positions)
            for i in dist_indices_sorted:
                x, y = i
                if history[y][0] not in object_ids and dist[x,y]<100:
                    if object_ids[x]==-1:
                        object_ids[x] = history[y][0]

            #Assigning new object IDs to objects that did not have any match
            for i in range(len(object_ids)):
                if object_ids[i]==-1:
                    self.obj_count+=1
                    object_ids[i] = self.obj_count

            history = list(zip(object_ids, curr_positions))
            return object_ids, history
        
        
#function to get angle offsets
def get_angle_offsets(x, y, resolution):
    im_c_x = resolution[0]/2
    im_c_y = resolution[1]/2
    
    move_x = x - im_c_x
    move_y = y - im_c_y
    
    move_x /=resolution[0]
    move_y /=resolution[1]
    
    return move_x, move_y
    
    
#Returns euclidean distance between two points
def e_dist(a,b):
    dist = 0
    for i in range(2): #Uses only frist two values of each point for calculating the distance
        dist += (a[i]-b[i])**2
    dist = dist**0.5
    return dist


def main(path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', './best.pt')
    
    vid = cv2.VideoCapture(path)
    
    tracker = Tracker()
    
    #Get color frame
    ret, frame = vid.read()
    
    resolution = list((frame.shape[:2]))
    resolution.reverse()
    
    tracker_history = {}
    
    while(ret):
        
        #obtain object detections from yolo model
        res = model(frame)
        
        curr_positions = []
        
        for i in res.pred[0]:
            i = i.detach().cpu().numpy()
            x1, y1, x2, y2, conf, c = i
            if conf<0.5:
                continue
                
            curr_positions.append([int((x1+x2)/2), int((y1+y2)/2), int(x2-x1), int(y2-y1)])

        #get object tracking results and assign object ID to each object
        if(len(curr_positions)!=0):  
            object_ids, new_tracker_history = tracker.track_objs(curr_positions, tracker_history)
            if len(new_tracker_history)!=0:
                tracker_history = new_tracker_history

            
        
        nearest_object = None
        max_area = 0
        
        
        #Drawing bounding boxes for detected objects
        for i in range(len(curr_positions)):               
            x,y,w,h = curr_positions[i]
            x1, y1, x2, y2 = int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2)
            if w*h > max_area:
                nearest_object=i
                max_area=w*h
                
            cv2.rectangle(frame, (int(x1),int(y1-15)), (int(x1)+30,int(y1)-30), (255,255,255), -1)
            cv2.putText(frame,  str(object_ids[i]), (x1, y1-15), cv2.FONT_HERSHEY_PLAIN,1, (255,0,0))
            frame = cv2.rectangle(frame, (int(x1),int(y1)), (int(x1+w),int(y1+h)), (0,255,0), 1)
       
    
        cv2.line(frame, (int(resolution[0]/2), int(resolution[1]/2 - 20)), (int(resolution[0]/2), int(resolution[1]/2 + 20)), (255,0,0), 1)
        cv2.line(frame, (int(resolution[0]/2 -20 ), int(resolution[1]/2)), (int(resolution[0]/2 + 20), int(resolution[1]/2)), (255,0,0), 1)
        
        
        #Determing nearest object to target for attack. Drawing the offset on view
        if nearest_object!=None:
            move_x, move_y = get_angle_offsets(curr_positions[nearest_object][0], curr_positions[nearest_object][1], resolution)
            cv2.putText(frame, str("Angle offsets: ({},{})".format(round(move_x,2),round(move_y,2))), (10, resolution[1]-25), cv2.FONT_HERSHEY_PLAIN,1, (255,0,0))
            move_x*=resolution[0]
            move_y*=resolution[1]
            x, y, w, h = curr_positions[nearest_object]
            start_point = (int(resolution[0]/2), int(resolution[1]/2))
            end_point = (int(resolution[0]/2 + move_x ), int(resolution[1]/2 + move_y))
            cv2.line(frame, start_point, end_point , (255,0,0), 1)
            
            cv2.line(frame, (x, y-4), (x, y+4), (0,0,255), 1)
            cv2.line(frame, (x-4, y), (x+4, y), (0,0,255), 1)
        
        cv2.imshow('video', frame)
        
        if cv2.waitKey(1)==ord('q'):
            break
            
        ret, frame = vid.read()
        
    vid.release()
    
if __name__=="__main__":
    main("./vid.mp4")