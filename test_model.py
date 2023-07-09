import cv2
import numpy as np
from CNN import CNN
from DataSet import *
import os
import sys

cwd = os.getcwd()


image_size=40


class Tens:

    def __init__(self):
        f1=open('data_set_config.txt','r')
        st=f1.readline()
        st1=st.split(' ')
        global image_size
        image_size=int(st1[0])
        shape=[int(st1[0]),int(st1[1]),int(st1[2])]
        print ('dataset image shape:',st)

        class_cnt=int(f1.readline())

        self.class_name=[]

        for i in range(0,class_cnt):
            self.class_name.append(f1.readline().replace('\n',''))
        f1.close()



        print ('initializing CNN...')

        self.cnn=CNN(shape,class_cnt)

        print ('CNN initialized !')
        print ('loading model...')
        self.cnn.load('model')
        print ('model loaded')


    def detect(self,src):
        global prev
        sr=[]
        src = cv2.resize(src,(image_size,image_size), interpolation = cv2.INTER_CUBIC)
        src=src.reshape([-1])
        #print src.shape
        sr.append(src)
        pred=self.cnn.predict(sr)
        return self.class_name[pred[0]]

tens=Tens()

dir='train'
dirlist=os.listdir(cwd+"/test/")

total_image=0
matched=0

for dir in dirlist:

    flist=os.listdir(cwd+"/test/"+dir)
    for filename in flist:
        img=cv2.imread(cwd+"/test/"+dir+"/"+filename,0)
        res=tens.detect(img)
        total_image+=1
        if res==dir:
            #print ('matched','actual',dir,'found',res)
            matched+=1
        else:
            print (dir,filename)
            print ('not matched','actual',dir,'found',res)
accuracy=float(matched)/float(total_image)
print ('acuracy',accuracy)
