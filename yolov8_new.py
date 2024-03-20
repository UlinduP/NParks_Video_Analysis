import numpy as np
import pandas as pd
import cv2 as cv
import torch
from ultralytics import YOLO
import google.generativeai as genai
import os
#import PIL.Image
from PIL import Image



safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]



genai.configure(api_key="AIzaSyCqTVcWQwa7gLneUfil-Pi6eHLfmvZcS-s")

model = YOLO("yolov8l.pt")
gem_model=genai.GenerativeModel('gemini-pro-vision',safety_settings=safety_settings)
gem_model.temperature=0.01

chat_model=genai.GenerativeModel("gemini-pro",safety_settings=safety_settings)
chat_model.temperature=0.01

vid=cv.VideoCapture("lifting.mp4")

if not vid.isOpened():
    print("Cant open the file")
    exit()

count=0
frame_no=0
fin_response=""
while True:
    ret,frame=vid.read()
    if not ret:
        break
    count+=1

    if count%20!=0:
        frame_no+=1
        continue
    human=False
    result=model.predict(frame)
    b=torch.unsqueeze(result[0].boxes.cls,1)
    c=torch.unsqueeze(result[0].boxes.conf,1)
    a=torch.cat((result[0].boxes.xyxy,c,b),1)
    tab=pd.DataFrame(a).astype("float")
    #print("leon")
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
            #cv.rectangle(frame,(x,y),(x1,y1),(0,255,0),2)
            human=True
            #print("leohwuieg")
            #print(x,y,x1,y1,(x1-x)*(y1-y))
    cv.imshow("Video",frame)
    img_pil = Image.fromarray(frame)
    #array1 = np.array(list1, dtype=np.uint8)
    #img_pil = Image.fromarray(array1)
    #print(list1)
    if human:
        response = gem_model.generate_content(["Is there shop lifting",img_pil])
        print(response.text)
        response.resolve()
        fin_response+="frame "+str(frame_no)+" : " + response.text + " \n"
        #fin_response+= response.text
    if cv.waitKey(0) & 0xFF==ord('q'):
        break

chat = chat_model.start_chat(history=[])
print("**********************************************************************")
print(fin_response)
chat_response=chat.send_message("This are image discriptions of a video understand and give a summarise by indicating the frame numbers: "+fin_response,stream=True)
chat_response.resolve()
print("**************************************************************************  summarising")
print(chat_response.text)
print("**********************************************************************")


vid.release()
cv.destroyAllWindows()
    
