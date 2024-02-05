import numpy as np
import pandas as pd
import cv2 as cv
import torch
from ultralytics import YOLO
import google.generativeai as genai
import os
#import PIL.Image
from PIL import Image
genai.configure(api_key="AIzaSyCqTVcWQwa7gLneUfil-Pi6eHLfmvZcS-s")

model = YOLO("yolov8l.pt")
gem_model=genai.GenerativeModel('gemini-pro-vision')

vid=cv.VideoCapture("vid1.mov")

if not vid.isOpened():
    print("Cant open the file")
    exit()

count=0

while True:
    ret,frame=vid.read()

    if not ret:
        break
    count+=1

    if count%3!=0:
        continue

    result=model.predict(frame)
    b=torch.unsqueeze(result[0].boxes.cls,1)
    c=torch.unsqueeze(result[0].boxes.conf,1)
    a=torch.cat((result[0].boxes.xyxy,c,b),1)
    tab=pd.DataFrame(a).astype("float")
    #print(tab)
    #print(f"the result is {result[0].boxes.cls}")
    list1=[]
    for index,row in tab.iterrows():
        if row[5]==0:
            #print("human")
            x=int(row[0])
            y=int(row[1])
            x1=int(row[2])
            y1=int(row[3])
            #crop=frame[y:y1,x:x1]
            #print(type(crop))
            #crop=cv.resize(crop,((y1-y)*2,(x1-x)*2))
            #cv.imshow("video",crop)
            #img_pil = Image.fromarray(crop)
            #list1.append(img_pil)
            cv.rectangle(frame,(x,y),(x1,y1),(0,255,0),2)
    cv.imshow("Video",frame)
    img_pil = Image.fromarray(frame)
    #array1 = np.array(list1, dtype=np.uint8)
    #img_pil = Image.fromarray(array1)
    #print(list1)
    response = gem_model.generate_content(["look the whole picturw and discribe the people which are drawn a green rectangle , there appearence and gender accurately",img_pil])
    print(response.text)
    if cv.waitKey(0) & 0xFF==ord('q'):
        break

vid.release()
cv.destroyAllWindows()
    
