from PIL import Image, ImageDraw
import face_recognition
import pickle
import glob
import os

known_faces = [
    # ['Lisa','lisa.jpg'],
    # ['Jennie','jennie.jpg'],
#     ['Rose','rose.jpg'],
#     ['Jisoo','jisoo.jpg'],
#     ['Sattawat','sattawat.png'],
#     ['Aaron','Aaron_Peirsol_0002.jpg','Aaron_Peirsol_0003.jpg','Aaron_Peirsol_0004.jpg']
]

# known_faces=[]

ROOT_FOLDER ="C:/Users/asus/Desktop/mainprojectv1/mainproject/trainpeople"

known_faces_names = []
known_faces_encodings = []
for path in glob.iglob(os.path.join(ROOT_FOLDER, "**", "*.jpg")):
    path = path.replace("\\","/")
    person = path.split("/")[-2]
    known_faces.append([person,path])
# print(known_faces)
for face in known_faces:
    #ถอดรหัส
    known_faces_names.append(face[0])
    print(face[0])
    face_image = face_recognition.load_image_file(face[1])
    print(face[1])
    try:
        face_encoding = face_recognition.face_encodings(face_image)[0] 
        known_faces_encodings.append(face_encoding) 
        print(known_faces_encodings)
    except:
        continue
# print(known_faces_names)
# print(face_encoding)
# print(face_encoding)
# print(known_faces_encodings)
pickle.dump((known_faces_names,known_faces_encodings),open('faces.p','wb'))
# for face in known_faces:
#     # print(face[0])
#     #ถอดรหัส
#     known_faces_names.append(face[0])
#     face_image = face_recognition.load_image_file(face[1])
#     face_encoding = face_recognition.face_encodings(face_image)[0]
#     known_faces_encodings.append(face_encoding)
    # print(face_encoding)
# print(known_faces_names)
# print(known_faces_encodings)
# print(face_encoding)
