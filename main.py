import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

# Preditor model instance creation
model=YOLO('yolov8s.pt')

# Run the model on the GPU instead of CPU
model.to("cuda")

# Areas of interest coordinates
area1=[(312,388),(289,390),(474,469),(497,462)]

area2=[(279,392),(250,397),(423,477),(454,469)]

# Function to get the coordinated of the mouse cursor when
# the cursor is inside the RGB window
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture('peoplecount1.mp4')

# Classes of interest list creation
my_file = open("classes.txt", "r")
class_list = my_file.read().split("\n")

# print(class_list)

count=0

# Run until Esc key is pressed
while True:    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 2 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))
#    frame=cv2.flip(frame,1)
    
    # Store the predicted objects data into a
    # Pandas dataframe
    results=model.predict(frame)
    # print(results)   
    a=results[0].boxes.data
    px=pd.DataFrame(a.cpu().numpy()).astype("float")
    # print(px)
    list=[]
    
    # If the detected object is a person,
    # draw a rectangle around them
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        obj_class=int(row[5])
        c=class_list[obj_class]
        if c == "person":
           cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
           cv2.putText(frame,str(c),(x1,y1),cv2.FONT_HERSHEY_COMPLEX,(0.5),(255,255,255),1)
        
      
            
            
        
    cv2.polylines(frame,[np.array(area1,np.int32)],True,(255,0,0),2)
    cv2.putText(frame,str('1'),(504,471),cv2.FONT_HERSHEY_COMPLEX,(0.5),(0,0,0),1)

    cv2.polylines(frame,[np.array(area2,np.int32)],True,(255,0,0),2)
    cv2.putText(frame,str('2'),(466,485),cv2.FONT_HERSHEY_COMPLEX,(0.5),(0,0,0),1)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()

