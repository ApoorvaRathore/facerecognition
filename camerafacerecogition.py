import dlib
import cmake
import face_recognition as fr
import cv2

f=cv2.CascadeClassifier('C:/Users/hp/Desktop/Techienest/haarcascade_frontalface_default.xml')
m_test=0
train_image=fr.load_image_file('C:/Users/hp/Downloads/008-2.jpg')
m_train=fr.face_encodings(train_image)[0]
#print(train_image)
#print(m_train)
#m_train=list(m_train)
print()
i=cv2.VideoCapture(0)
while(1):
    
    return_value, img1 = i.read()
    #img1=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    l=f.detectMultiScale(img1,1.3,7)
    print(l)
    if(len(l)>0):
        for (x,y,w,h) in l:     #x,y coordinates of top left corner
            cv2.rectangle(img1,(x,y),(x+w,y+h),(0,0,255),10)  #x+w,y+h coordinates of bottom right corner
            cv2.imshow("Video",img1)
            m_test=[]
            m_test=fr.face_encodings(img1)[0]
           # print(m_test)



#f=2


            x=fr.compare_faces([m_train],m_test)
            print(x)
            if(x[0]==True):
                 font = cv2.FONT_HERSHEY_SIMPLEX
                 cv2.putText(img1,'Apoorva',(500,500), font, 4,(0,255,0),2,cv2.LINE_AA)
                 
            else:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img1,'Unknown',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)
                #cv2.label("Unknown")
    z=cv2.waitKey(1)
    if(z==ord('q')):
        #player.stop()
        break

    
i.release()
cv2.destroyAllWindows()
