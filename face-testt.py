#ตัวทอดลอง face-recognition 28/4/2022!!!!!!!!!
import importlib
from PIL import Image, ImageDraw
import face_recognition
import numpy as np
import pickle
import os
import glob
import cv2
from datetime import date, datetime



datatext = []
# path = glob.glob("D:/project/test1/Flaskmyweb/testpeople/*.jpg")
# C:/xampp/htdocs/2/skydash-free-bootstrap-admin-template-main/template/pages/tables/test2/nomask2022-11-14 16-49-09.jpg

path = glob.glob("C:/Users/asus/Desktop/mainprojectv1/mainproject/test2/t10.jpg")
# path = glob.glob("C:/Users/asus/Desktop/mainprojectv1/mainproject/trainpeople/time/313880529_479847607459941_5006830136515904568_n.jpg")
for file in path:
    known_face_names, known_face_encodings = pickle.load(open('faces.p','rb'))
        
    image = Image.open(file)
    face_locations = face_recognition.face_locations(np.array(image))
    face_encodings = face_recognition.face_encodings(np.array(image), face_locations)
    draw = ImageDraw.Draw(image)
    # print(file)
    # แสดงรูปที่อ่านได้จากในpath
    # img = cv2.imread(file)
    # cv2.imshow("img",img)
    image.show()
    # print(face_encodings)
    # print(face_locations)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print("l")
    pathnomask  = r'test2/'
    for face_encoding , face_location in zip(face_encodings, face_locations):
        face_distances = face_recognition.face_distance(known_face_encodings,face_encoding)
        best_match_index = np.argmin(face_distances)
        print(face_distances)
        print(best_match_index)
        if (face_distances < 0.9).all():
                name = known_face_names[best_match_index]
                top, right, bottom, left = face_location
                # img_temp = image[right:right+top, bottom:bottom+top]
                draw.rectangle([left,top,right,bottom])
                draw.text((left,top), name)
                datatext.append(name) 
                text_file = open("report.txt", "w")
                n = text_file.write(name)
                text_file.close()
                image.show()


                
                print(best_match_index)
                # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                # eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
                # # img = cv2.imread(path)
                # for file in path:
                #     # img = Image.open(file)
                #     img = cv2.imread(file)
                #     # draw = ImageDraw.Draw(img)
                #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                #     faces = face_cascade.detectMultiScale(gray, 1.301, 5)
                #     # print("Found {0} faces".format(len(faces)))
                #     for (x,y,w,h) in faces:
                #         cv2.rectangle(img,(x,y),(x+w,y+h),(100,200,250),2)
                #         roi_gray = gray[y:y+h, x:x+w]
                #         roi_color = img[y:y+h, x:x+w]
                #         img_temp = img[y:y+h, x:x+w]
                #         curr_datetime = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
                #         img_name = "{}{}{}.jpg".format("nomask",name,curr_datetime)
                #         cv2.imwrite(os.path.join(pathnomask,img_name),img_temp)     
        else:
            name = "unknow"
            top, right, bottom, left = face_location
            draw.rectangle([left,top,right,bottom])
            draw.text((left,top), name)
            image.show()
        # print(face_distances)
        # print(name) 
            datatext.append(name) 
print(datatext)
file = open("report.txt", "w+")
content = str(datatext)
file.write(content)
file.close()