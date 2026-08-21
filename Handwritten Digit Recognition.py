#!/usr/bin/env python
# coding: utf-8

# 1. Screen Capture

# In[11]:


get_ipython().system('pip install pyscreenshot')


# In[12]:


import pyscreenshot as ImageGrab
import time


# In[13]:


pwd


# from PIL import Image
# 
# # Assuming you have an image object or path to an image file
# # For example, if you're loading an image from a file:
# image_path = "C:\Users\hp\captured_images"
# im = Image.open(image_path)
# 
# # Save the image
# im.save(images_folder + str(i) + '.png')
# 

# images_folder="captured_images/0/"
# for i in range(0,5):
#     time.sleep(8)
#     ImageGrab.grab(bbox=(60,170,400,550))  #cordinates x1,y1, x2,y2
#     print("saved......",i)
#     im.save(images_folder+str(i)+'.png')
#     print("clear screen now and redrew now....")

# In[1]:


import time
from PIL import ImageGrab

images_folder = "captured_images/0/"

for i in range(1, 200):
    time.sleep(8)
    im = ImageGrab.grab(bbox=(60, 250, 400, 650))  # coordinates x1, y1, x2, y2
    print("saved......", i)
    im.save(images_folder + str(i) + '.png')
    print("clear screen now and redraw now....")


# In[3]:


import time
from PIL import ImageGrab

images_folder = "captured_images/1/"

for i in range(0, 200):
    time.sleep(8)
    im = ImageGrab.grab(bbox=(60, 250, 400, 650))  # coordinates x1, y1, x2, y2
    print("saved......", i)
    im.save(images_folder + str(i) + '.png')
    print("clear screen now and redraw now....")


# In[4]:


import time
from PIL import ImageGrab

images_folder = "captured_images/2/"

for i in range(0, 200):
    time.sleep(8)
    im = ImageGrab.grab(bbox=(60, 250, 400, 650))  # coordinates x1, y1, x2, y2
    print("saved......", i)
    im.save(images_folder + str(i) + '.png')
    print("clear screen now and redraw now....")


# In[8]:


import time
from PIL import ImageGrab

images_folder = "captured_images/3/"

for i in range(150, 200):
    time.sleep(5)
    im = ImageGrab.grab(bbox=(60, 250, 400, 650))  # coordinates x1, y1, x2, y2
    print("saved......", i)
    im.save(images_folder + str(i) + '.png')
    print("clear screen now and redraw now....")


# In[21]:


import time
from PIL import ImageGrab

images_folder = "captured_images/8/"

for i in range(150, 200):
    time.sleep(5)
    im = ImageGrab.grab(bbox=(60, 250, 400, 650))  # coordinates x1, y1, x2, y2
    print("saved......", i)
    im.save(images_folder + str(i) + '.png')
    print("clear screen now and redraw now....")


# In[25]:


import time
from PIL import ImageGrab

images_folder = "captured_images/9/"

for i in range(116, 117):
    time.sleep(5)
    im = ImageGrab.grab(bbox=(60, 250, 400, 650))  # coordinates x1, y1, x2, y2
    print("saved......", i)
    im.save(images_folder + str(i) + '.png')
    print("clear screen now and redraw now....")


# In[14]:


def one_time():
    import time
    from PIL import ImageGrab

    images_folder = "captured_images/9/"

    for i in range(150, 200):
        time.sleep(8)
        im = ImageGrab.grab(bbox=(60, 250, 400, 650))  # coordinates x1, y1, x2, y2
        print("saved......", i)
        im.save(images_folder + str(i) + '.png')
        print("clear screen now and redraw now....")


# Generate Dataset

# In[8]:


get_ipython().system('pip install opencv-python')
get_ipython().system('pip install pandas')
get_ipython().system('pip install scikit-learn')


# In[15]:


import cv2
import csv
import glob
  
header  =["label"]
for i in range(0,784):
   header.append("pixel"+str(i))
with open('dataset.csv', 'a') as f:
   writer = csv.writer(f)
   writer.writerow(header)
    
for label in range(10):
   dirList = glob.glob("captured_images/"+str(label)+"/*.png")
    
   for img_path in dirList:
       im= cv2.imread(img_path)
       im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
       im_gray = cv2.GaussianBlur(im_gray,(15,15), 0)
       roi= cv2.resize(im_gray,(28,28), interpolation=cv2.INTER_AREA)
        
       data=[]
       data.append(label)
       rows, cols = roi.shape
        
       ## Fill the data array with pixels one by one.
       for i in range(rows):
           for j in range(cols):
               k =roi[i,j]
               if k>100:
                   k=1
               else:
                   k=0
               data.append(k)
       with open('dataset.csv', 'a') as f:
           writer = csv.writer(f)
           writer.writerow(data)


# Load the dataset

# In[16]:


import pandas as pd
from sklearn.utils import shuffle
data=pd.read_csv('dataset.csv')
data


# In[17]:


get_ipython().system('pip install matplotlib')
get_ipython().system('pip install joblib')


# Separation of dependent and independent variable

# In[18]:


X = data.drop(["label"], axis=1)
Y=data["label"]


# Preview of one image using matplotlib

# In[19]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import cv2
idx = 716
img = X.loc[idx].values.reshape(28,28)
print(Y[idx])
plt.imshow(img)


# Train Test split

# In[7]:


from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(X,Y, test_size = 0.25)


# Fit the model using svc and also save the model using joblib
# 
# 

# In[8]:


import joblib
from sklearn.svm import SVC
classifier=SVC(kernel="linear", random_state=6)
classifier.fit(train_x,train_y)
joblib.dump(classifier, "model/digit_recognizer")


# Calculate Accuracy

# In[9]:


from sklearn import metrics
prediction=classifier.predict(test_x)
print("Accuracy= ",metrics.accuracy_score(prediction, test_y))


# Prediction of image drawn in paint

# In[10]:


import joblib
import cv2
import numpy as np #pip install numpy
import time
import pyscreenshot as ImageGrab
  
model=joblib.load("model/digit_recognizer")
images_folder="img/"

while True:
   img=ImageGrab.grab(bbox=(60, 250, 400, 650))
    
   img.save(images_folder+"img.png")
   im = cv2.imread(images_folder+"img.png")
   im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
   im_gray  =cv2.GaussianBlur(im_gray, (15,15), 0)
    
   #Threshold the image
   ret, im_th = cv2.threshold(im_gray,100, 255, cv2.THRESH_BINARY)
   roi = cv2.resize(im_th, (28,28), interpolation  =cv2.INTER_AREA)
    
   rows,cols=roi.shape
    
   X = []
    
   ##  Fill the data array with pixels one by one.
   for i in range(rows):
       for j in range(cols):
           k = roi[i,j]
           if k>100:
               k=1
           else:
               k=0
           X.append(k)
            
   predictions = model.predict([X])
   print("Prediction:",predictions[0])
   cv2.putText(im, "Prediction is: "+str(predictions[0]), (20,20), 0, 0.8,(0,255,0),2,cv2.LINE_AA)
    
   cv2.startWindowThread()
   cv2.namedWindow("Result")
   cv2.imshow("Result",im)
   cv2.waitKey(10000)
   if cv2.waitKey(1)==13: #27 is the ascii value of esc, 13 is the ascii value of enter
       break
cv2.destroyAllWindows()


# In[ ]:




