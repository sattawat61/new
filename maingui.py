# ส่วนชอง GUI
# Import required Libraries
from tkinter import *
from PIL import Image, ImageTk
import cv2
from counting.centroidtracker import CentroidTracker
from counting.trackableobject import TrackableObject
import numpy as np
import time
from datetime import date, datetime
from tkinter import ttk
import tkinter as tk
from tkinter import messagebox
import os
import webbrowser

from config import *
import pymysql
import glob
import face_recognition
import pickle
from PIL import Image, ImageDraw
import shutil

con = pymysql.connect(HOST,USER,PASS,DATABASE)


webcam =cv2.VideoCapture(0)
# webcam =cv2.VideoCapture("C:/Users/asus/Desktop/mainprojectv1/mainproject/record/2022-11-11 14-10-00.avi")


curr_datetime = datetime.now().strftime('%Y-%m-%d %H-%M-%S')

# frame_width = int(webcam.get(3))
# frame_height = int(webcam.get(4))
frame_width = 800
frame_height = 600
frame_size = (frame_width,frame_height)
output = cv2.VideoWriter('./record/{}.avi'.format(curr_datetime), cv2.VideoWriter_fourcc('M','J','P','G'),5,frame_size)
# webcam =cv2.VideoCapture(0)
# fourcc=cv2.VideoWriter_fourcc(*'XVID')
# out=cv2.VideoWriter('new.avi',fourcc,20.0,(640,480))

success2, image = webcam.read()

count_frame = 0 #ตัวนับ frame 

# global lista,i,o,check,product,sum_hand,listdata_hand,add_hand,numBox
global i
# check = 0
# lista = []
# listdata_hand = []
# product = []
i = 0
# o = 0 
# succ = 0
# sum_hand = 1
sum_people = 1
# add_hand = 0
numBox = 0

global root
root = Tk()
root.title("Mask Detection")
# กำหนดขนาดและตำแหน่งหน้าจอ
root.geometry("500x400+700+300")
root.config(bg="#F5F7FF")

global fps_end_time,fps_start_time
name_var = tk.StringVar()
num_roi = tk.StringVar()
name_Pro = tk.StringVar()
Prodlist = []
fps_end_time = 0
fps_start_time = 0

# global Sq_people,SQpeople,dot_people,dotpeople

# Sq_people = tk.StringVar()
# SQpeople = "close"

# dot_people = tk.StringVar()
# dotpeople = "close"

# id_people = tk.StringVar()
# IDpeople = "close"

############################################# People Detect #############################################

confThreshold = 0.5
#confidence ของกรอบนั้นสูงกว่า 0.5 หรือไม่ ถ้าต่ำกว่าก็ไม่ display
nmsThreshold = 0.4 
#โดยหลักการคำนวณคือ หาส่วนที่ Intersect  
inpWidth = 320    
inpHeight = 320      

# pathnomask = r'C:/xampp/htdocs/skydash-free-bootstrap-admin-template-main/template/pages/tables/test/'
pathnomask = r'C:/xampp/htdocs/2/skydash-free-bootstrap-admin-template-main/template/pages/tables/test/'
pathnomask2 = r'C:/xampp/htdocs/2/skydash-free-bootstrap-admin-template-main/template/pages/tables/test2/'
pathnomasksave = r'test/'
pathnomasksave2 = r'test2/'

classesFile = "C:/Users/asus/Desktop/mainprojectv1/mainproject/weight/coco.names";
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

modelConfiguration = "C:/Users/asus/Desktop/mainprojectv1/mainproject/weight/yolov4-custom.cfg"; 
modelWeights = "C:/Users/asus/Desktop/mainprojectv1/mainproject/weight/yolov4-custom_best.weights"; 
print("[INFO] loading model...")
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

writer = None
 
W = None
H = None 

ct = CentroidTracker(maxDisappeared=2, maxDistance=100)
trackers = []
trackableObjects = {}

global ay_count
ay_count = []

totalDown = 0
totalUp = 0
totalAll = 0

def getOutputsNames(net):
    layer_names = net.getLayerNames()
    return [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

def drawPred(classId, conf, left, top, right, bottom):
    # วาดกรอบคน
    if classId == 0:
        name = "mask"
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left , top - 6), font, 1.0, (0, 255, 0), 1)
    else :
        name = "nomask"
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left , top - 6), font, 1.0, (0, 0, 255), 1)
    # frameHeight = frame.shape[0]
    # frameWidth = frame.shape[1]

def postprocess(frame, outs):
    #เก็บขนาดเก็บค่าความสูงของตัวแปร
    global ay_count,classIds,box
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    rects = []
    classIds = []
    confidences = []
    boxes = []
    #วนค่าเพื่อนำไปวาดกรอบเเล้วเก็บid
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)

                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    #วนค่านับกรอบที่ได้เเล้วใช้ฟังก์ชั่นนับคนเข้าออก
    for i in indices:
        i = i 
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]

        if classIds[i] == 0 or classIds[i] == 1:
            rects.append((left, top, left + width, top + height))
            objects = ct.update(rects)
            counting(objects)
            drawPred(classIds[i], confidences[i], left, top, left + width, top + height)

def counting(objects):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    img_copy = frame.copy()

    global totalDown
    global totalUp
    global totalAll
    global ay_count
    global sum_people,Tablae_People
    
    for (objectID, centroid) in objects.items():

        to = trackableObjects.get(objectID, None)
 
        if to is None:
            to = TrackableObject(objectID, centroid)

        else:
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)
            # ขาออก
            if not to.counted:
                # if direction < 0 and centroid[1] in range(frameHeight//2 - 150, frameHeight//2 + 150):
                if direction < 0 and centroid[1] in range(frameHeight//2 - 90, frameHeight//2 + 75):
                    totalUp += 1
                    dt = datetime.now()
                    date = str(dt.year)+"-"+ str(dt.month)+"-"+ str(dt.day)
                    time = str(dt.hour)+":"+ str(dt.minute)+":"+ str(dt.second)
                    ay_count.append([totalAll,date,time,'Out'])
                    to.counted = True

                    if classIds[i] == 0:
                        print("out mask")
                        Tablae_People.insert(parent='',index='end',iid=sum_people,text='',
                        values=(sum_people,date,time,'Out','mask'))
                        Tablae_People.see(sum_people)
                        sum_people = sum_people + 1
                        img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
                        img_temp = img_copy[int(box[1]):int(box[1]) + int(box[3]), int(box[0]):int(box[0]) + int(box[2]), :]
                        curr_datetime = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
                            # img_name = "{}{}.jpg".format("nomask",curr_datetime)
                            # cv2.imwrite(os.path.join(pathnomask,img_name),img_temp)
                        with con:
                            name = "unknow"
                            cur = con.cursor()
                            sql = "insert into tb_member (usr_name,usr_date,usr_time,usr_status,usr_statusmask) VALUES (%s,%s,%s,%s,%s)"
                            cur.execute(sql,(name,date,time,"out","mask"))
                            con.commit()
                    if classIds[i] == 1:
                        print("out Nomask")
                        Tablae_People.insert(parent='',index='end',iid=sum_people,text='',
                        values=(sum_people,date,time,'Out','Nomask'))
                        Tablae_People.see(sum_people)
                        sum_people = sum_people + 1

                        with con:
                            name = "unknow"
                            cur = con.cursor()
                            sql = "insert into tb_member (usr_name,usr_date,usr_time,usr_status,usr_statusmask) VALUES (%s,%s,%s,%s,%s)"
                            cur.execute(sql,(name,date,time,"out","nomask"))
                            con.commit()
                        
                # ขาเข้า
                elif direction > 0 and centroid[1] in range(frameHeight//2 - 100, frameHeight//2 + 145):
                # elif direction > 0 and centroid[1] in range(frameHeight//2 - 90, frameHeight//2 + 75):
                    totalDown += 1
                    dt = datetime.now()
                    date = str(dt.year)+"-"+ str(dt.month)+"-"+ str(dt.day)
                    time = str(dt.hour)+":"+ str(dt.minute)+":"+ str(dt.second)
                    ay_count.append([totalAll,date,time,'In'])
                    to.counted = True
                    # if direction > 0 and centroid[1] in range(frameHeight//1-100, frameHeight//1-100):
                    # print("ok")
                    if classIds[i] == 0:
                        # if direction > 0 and centroid[1] in range(frameHeight//2+25, frameHeight//2+25):
                        print("in mask")
                        Tablae_People.insert(parent='',index='end',iid=sum_people,text='',
                        values=(sum_people,date,time,'in','mask'))
                        Tablae_People.see(sum_people)
                        sum_people = sum_people + 1
                    
                        img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
                        with con:
                            name = "unknow"
                            cur = con.cursor()
                            sql = "insert into tb_member (usr_name,usr_date,usr_time,usr_status,usr_statusmask) VALUES (%s,%s,%s,%s,%s)"
                            cur.execute(sql,(name,date,time,"in","mask"))
                            con.commit()
                            print("ttttttttttttttt")
                    if classIds[i] == 1:
                        # if direction < 0 and centroid[1] in range(frameHeight//2+25, frameHeight//2+25):
                        print("in Nomask")
                        Tablae_People.insert(parent='',index='end',iid=sum_people,text='',
                        values=(sum_people,date,time,'in','Nomask'))
                        Tablae_People.see(sum_people)
                        sum_people = sum_people + 1
                        img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
                        # img_temp = img_copy[int(box[1]):int(box[1]) + int(box[3]), int(box[0]):int(box[0]) + int(box[2]), :]
                        img_temp = img_copy
                        curr_datetime = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
                        img_name = "{}{}.jpg".format("nomask",curr_datetime)
                        cv2.imwrite(os.path.join(pathnomask,img_name),img_temp)
                        image = Image.open("C:/xampp/htdocs/2/skydash-free-bootstrap-admin-template-main/template/pages/tables/test/"+img_name) 
                        # image.show() 
                        known_face_names, known_face_encodings = pickle.load(open('faces.p','rb'))
                        face_locations = face_recognition.face_locations(np.array(image))
                        face_encodings = face_recognition.face_encodings(np.array(image), face_locations)
                        draw = ImageDraw.Draw(image)
                        # name="unknow"
                        # with con:
                        #     cur2 = con.cursor()
                        #     sql2 = "insert into tb_memberinmask (usr_name,usr_date,usr_time,usr_status,usr_statusmask VALUES (%s,%s,%s,%s,%s)"
                        #     cur2.execute(sql2,(name,date,time,"in","nomask"))  
                        #     con.commit()
                        for face_encoding , face_location in zip(face_encodings, face_locations):
                            face_distances = face_recognition.face_distance(known_face_encodings,face_encoding)
                            best_match_index = np.argmin(face_distances)
                            # print(face_distances)
                            # print(best_match_index)
                            if (face_distances < 0.6).any():
                                name = known_face_names[best_match_index]
                                top, right, bottom, left = face_location
                                    # img_temp = image[right:right+top, bottom:bottom+top]
                                draw.rectangle([left,top,right,bottom])
                                draw.text((left,top), name)
                                    # image.show()       
                                print(face_distances)
                                print(best_match_index)
                                # print(name)
                                # print(img_name)
                                with con:
                                    cur = con.cursor()
                                    sql = "insert into tb_member (usr_name,usr_date,usr_time,usr_status,usr_statusmask,usr_statusrec,usr_pic) VALUES (%s,%s,%s,%s,%s,%s,%s)"
                                    cur.execute(sql,(name,date,time,"in","nomask","yes","test/"+img_name))  
                                    con.commit()
                                               
                            else:
                                name = "unknow"
                                top, right, bottom, left = face_location
                                draw.rectangle([left,top,right,bottom])
                                draw.text((left,top), name)
                                print(face_distances)
                                print(best_match_index)
                                with con:
                                    cur = con.cursor()
                                    sql = "insert into tb_member (usr_name,usr_date,usr_time,usr_status,usr_statusmask,usr_statusrec,usr_pic) VALUES (%s,%s,%s,%s,%s,%s,%s)"
                                    cur.execute(sql,(name,date,time,"in","nomask","yes","test/"+img_name))  
                                    con.commit()
                                # with con:
                                #     cur = con.cursor()
                                #     sql = "insert into tb_memberinmask (usr_name,usr_date,usr_time,usr_status,usr_statusmask VALUES (%s,%s,%s,%s,%s)"
                                #     cur.execute(sql,(name,date,time,"in","nomask"))  
                                #     con.commit() 
                    
            
        trackableObjects[objectID] = to
        
        text = "ID {}".format(objectID)

# กลับหน้าแรก
def RunToMain():
    root.deiconify()
    win.destroy()

# detect
def run_detect():

    global frame,image,fps_end_time,fps_start_time,count_frame

    count_frame = count_frame + 1

    ###### FPS ######
    fps_end_time = time.time()
    time_diff = fps_end_time - fps_start_time
    fps = 1/(time_diff)
    fps_start_time = fps_end_time
    fps_text = "FPS: {: .2f}".format(fps)
    
    time_text = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    ##!!YOLO!!
    #### People Detect ######
    hasFrame, frame = webcam.read()
    frame = cv2.resize(frame, (800,600), interpolation = cv2.INTER_AREA)
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    cv2.rectangle(frame, (0,frameHeight//2 - 100), (frameWidth,frameHeight//2 + 145), (0, 255, 0), 2)
    cv2.rectangle(frame, (0,frameHeight//2 - 90), (frameWidth,frameHeight//2 + 75), (0, 0, 0), 2) 
    # # cv2.line(frame, (0, frameHeight // 2-50), (frameWidth, frameHeight // 2-50), (0, 255, 255), 2)
    # cv2.rectangle(frame, (0, frameHeight // 2-70), (frameWidth, frameHeight // 2+85), (0, 255, 255), 2)
    info = [("out", totalUp),("in", totalDown)]
    for (i, (k, v)) in enumerate(info):
        text = "{} : {}".format(k, v)
        cv2.putText(frame, text, (10, 50 - ((i * 20))),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 230), 2)

        blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

        net.setInput(blob)
    if(count_frame > 0):
        outs = net.forward(getOutputsNames(net))
        postprocess(frame, outs)
        count_frame = 0

    cv2.putText(frame, fps_text, (570, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),2)        
    cv2.putText(frame, time_text, (10, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),2)        
    cv2image2= cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    img2 = Image.fromarray(cv2image2)
    imgtk2 = ImageTk.PhotoImage(image = img2)
    label02.imgtk = imgtk2
    label02.configure(image=imgtk2)

    # save
    output.write(frame)

    win.after(10, run_detect)

# ส่วนฟังก์ชั่นเรียกใช้แมส
def detectionmask():
    # print('detectmask')
    root.withdraw()
    
    global ay_count,Tablae_People
    global win,label02
    win = Toplevel(root)
    win.geometry("1200x800+150+0")#655x590 #1305x590
    win.resizable(width=False, height=False)
    win.config(bg="#F5F7FF")
    label04 = Label(win, text = 'People Detect', font=('calibre 20 underline bold')).pack()
    label02 =Label(win)
    label02.pack()
    
    btn001 = Button(win,text="ย้อนกลับ", fg="white",command=RunToMain, font=('calibre',16), bg="#369F36", width=10)
    btn001.place(x = 50, y = 730)
    btn003 = Button(win,text="จบการทำงาน", fg="white",command=exitbtn, font=('calibre',16), bg="#B9062F", width=10)
    btn003.place(x = 1035, y = 730)#517x540
    
    run_detect()

    ####### People Table ######
    Tablae_People_scroll = Scrollbar(win)
    Tablae_People_scroll.place(x = 995, y = 640 , height=150)
    Tablae_People = ttk.Treeview(win,height=6)
    Tablae_People.configure(yscrollcommand=Tablae_People_scroll.set)
    Tablae_People_scroll.config(command=Tablae_People.yview)

    Tablae_People['columns'] = ('player_id', 'player_name', 'player_Rank', 'player_states','player_class')
    Tablae_People.column("#0", width=0,  stretch=NO)
    Tablae_People.column("player_id",anchor=CENTER, width=159)
    Tablae_People.column("player_name",anchor=CENTER,width=159)
    Tablae_People.column("player_Rank",anchor=CENTER,width=159)
    Tablae_People.column("player_states",anchor=CENTER,width=140)
    Tablae_People.column("player_class",anchor=CENTER,width=140)
    Tablae_People.heading("#0",text="",anchor=CENTER)
    Tablae_People.heading("player_id",text="No",anchor=CENTER)
    Tablae_People.heading("player_name",text="Date",anchor=CENTER)
    Tablae_People.heading("player_Rank",text="Time",anchor=CENTER)
    Tablae_People.heading("player_states",text="Status",anchor=CENTER)
    Tablae_People.heading("player_class",text="Status2",anchor=CENTER)

    Tablae_People.pack()
    


def reportwindow():
    webbrowser.open_new("http://localhost/2/skydash-free-bootstrap-admin-template-main/template/index.php")
    print('report') 

    datatext = []
    pictext = []
    path = glob.glob("C:/xampp/htdocs/2/skydash-free-bootstrap-admin-template-main/template/pages/tables/test/*.jpg")
    # path = glob.glob("C:/Users/asus/Desktop/mainprojectv1/mainproject/trainpeople/time/313880529_479847607459941_5006830136515904568_n.jpg")
    for file in path:
        known_face_names, known_face_encodings = pickle.load(open('faces.p','rb'))
            
        image = Image.open(file)
        face_locations = face_recognition.face_locations(np.array(image))
        face_encodings = face_recognition.face_encodings(np.array(image), face_locations)
        draw = ImageDraw.Draw(image)
  
        for face_encoding , face_location in zip(face_encodings, face_locations):
            face_distances = face_recognition.face_distance(known_face_encodings,face_encoding)
            best_match_index = np.argmin(face_distances)
            print(face_distances)
            # print(best_match_index)
            if (face_distances < 0.9).all():
                name = known_face_names[best_match_index]
                top, right, bottom, left = face_location
                    # img_temp = image[right:right+top, bottom:bottom+top]
                draw.rectangle([left,top,right,bottom])
                draw.text((left,top), name)
                datatext.append(name) 
                    # image.show()       
                print(best_match_index)  
            else:
                name = "unknow"
                top, right, bottom, left = face_location
                draw.rectangle([left,top,right,bottom])
                draw.text((left,top), name)
                    # image.show()
                # print(face_distances)
                # print(name) 
                datatext.append(name) 
        with con:
            cur = con.cursor()
            sql = "SELECT * FROM tb_member"
            cur.execute(sql)
            rows = cur.fetchall()

        for usr_id in range(len(datatext)):
            sql = "UPDATE tb_member SET usr_name = (%s),usr_statusrec = (%s)WHERE usr_id = (%s)"
            cur.execute(sql,(datatext[usr_id],"yes",usr_id+1))
            con.commit()
            print(pictext)
            
            print(datatext)

# ส่วนเรียกดูวิดิโอย้อนหลัง
def openrecord():
    webbrowser.open_new("C:/Users/asus/Desktop/mainprojectv1/mainproject/record")
    print('reccoed')

# ส่วนหน้าแอดมิน
def adminpage():
    webbrowser.open_new("http://localhost/2/skydash-free-bootstrap-admin-template-main/template/index.php")
    print('adminpage')

def recognize():
    known_faces = []


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
    pickle.dump((known_faces_names,known_faces_encodings),open('faces.p','wb'))
    tk.messagebox.showinfo("showinfo", "Finish")
    print('recognize')

# ปุ่มออกจากโปรแกรม
def exitbtn():
    confrim = tk.messagebox.askquestion("ยืนยันการปิดโปรแกรม","ต้องการปิดโปรแกรมหรือไหม ?",icon = 'warning')
    if confrim == "yes":
        root.destroy()

# ปุ่มบนหน้า GUI
btn1 = Button(root,text="Detection",width=20,height=2,font=20,bg="white",command=detectionmask).place(x=125,y=50)
btn3 = Button(root,text="Record",width=20,height=2,font=20,bg="white",command=openrecord).place(x=125,y=110)
btn4 = Button(root,text="Admin",width=20,height=2,font=20,bg="white",command=adminpage).place(x=125,y=170)
btn5 = Button(root,text="recognize",width=20,height=2,font=20,bg="white",command=recognize).place(x=125,y=230)
btn6 = Button(root,text="Exit",width=20,height=2,font=20,bg="white",command=exitbtn).place(x=125,y=290)
root.mainloop()

# save
webcam.release()
output.release()