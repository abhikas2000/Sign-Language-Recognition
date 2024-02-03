import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import mediapipe as mp
from math import dist
import pickle

folder=os.path.join("data")

mp_hands=mp.solutions.hands
mp_drawing=mp.solutions.drawing_utils
mp_drawing_styles=mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.7, max_num_hands=1)

def distance_calculation(temp):
  res=[]
  landmark_set=[(0,8),(0,12),(0,16),(0,20),(0,4),(1,4),(5,8),(9,12),(13,16),(17,20),(8,12),(8,16),(8,20),(4,8),(4,20),(4,12),(4,6),(4,10),(4,14),(4,18)]
  for m,n in landmark_set:
    res.append(dist(temp[m],temp[n]))
  return np.array(res)

data={}
for dir in os.listdir(folder):
    img = cv2.imread(os.path.join(folder, dir, '0.jpg'))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results=hands.process(img_rgb)

    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
         myHand = results.multi_hand_landmarks[0]
         temp={}
         for id, lm in enumerate(myHand.landmark):
            height, width, _ = img.shape
            temp[id]=[lm.x,lm.y]
         temp_y=distance_calculation(temp)
         temp_y/=max(temp_y)
         data[chr(int(dir)+65)]=temp_y

with open('rms_model.pickle', 'wb') as f:
    pickle.dump(data, f)
f.close()


fig,ax=plt.subplots(5,2,figsize=(20,20))
a,b=0,0
for i in range(8):
    l1,l2,l3=chr(65+3*i),chr((65+3*i)+1), chr((65+3*i)+2)
    ax[a][b].plot(data[l1],color="red",label=l1)
    ax[a][b].plot(data[l2],color="blue",label=l2)
    ax[a][b].plot(data[l3],color="green",label=l3)
    ax[a][b].legend(loc='upper right')
    if b==0:
        b=1
    else:
        b=0
        a+=1
ax[4][0].plot(data['Y'],color="red",label='Y')
ax[4][0].plot(data['Z'],color="blue",label='Z')
ax[4][0].legend(loc='upper right')
plt.show()