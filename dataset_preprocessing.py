import cv2
import numpy as np
import numpy as np
import cv2
import imutils
import glob

number=0
none= 0
label=0
imgs=[]
inner_circle=[]


for filepath in glob.iglob('UBIRIS_800_600/Sessao_1/*'):
    
    number= 0
    for filefilepath in glob.iglob(filepath+'/*'):
        if filefilepath[-1] == 'g':

            img	= cv2.imread(filefilepath)
            #gray_img	=	cv2.cvtColor(planets,	cv2.COLOR_BGR2GRAY)
            #img	= cv2.medianBlur(gray_img,	5)
            imgs.append(img)
            img=cv2.resize(img,(200,150))
            cv2.imwrite('final_image/'+str(label)+'.'+str(number)+'.jpg',img)
            number=number+1
    label=label+1

