# Import required Libraries
from tkinter import *
from PIL import Image, ImageTk
import cv2
from counting.centroidtracker import CentroidTracker
from counting.trackableobject import TrackableObject
import numpy as np
import time
from datetime import datetime
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

con = pymysql.connect(HOST,USER,PASS,DATABASE)


webcam =cv2.VideoCapture("C:/Users/User/Desktop/mainproject/data/video/test8.mp4")
# webcam =cv2.VideoCapture(0)
success2, image = webcam.read()

count_frame = 0 #ตัวนับ frame 

global lista,i,o,check,product,sum_hand,listdata_hand,add_hand,numBox
check = 0
lista = []
listdata_hand = []
product = []
i = 0
o = 0 
succ = 0
sum_hand = 1
sum_people = 1
add_hand = 0
numBox = 0

global root
root = Tk()
root.title("วิเคราะห์การเลือกซื้อด้วยเวลา")
root.geometry("300x330+500+300")
root.resizable(width=False, height=False)

global name_var,num_roi,fps_end_time,fps_start_time,name_Pro,Prodlist
name_var = tk.StringVar()
num_roi = tk.StringVar()
name_Pro = tk.StringVar()
Prodlist = []
fps_end_time = 0
fps_start_time = 0

global Sq_people,SQpeople,dot_people,dotpeople

Sq_people = tk.StringVar()
SQpeople = "close"

dot_people = tk.StringVar()
dotpeople = "close"

id_people = tk.StringVar()
IDpeople = "close"

############################################# People Detect #############################################

confThreshold = 0.5
nmsThreshold = 0.4   
inpWidth = 320    
inpHeight = 320      

pathnomask = r'test/'

# classesFile = "classes.names";
classesFile = "coco.names";
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# modelConfiguration = "yolov3_testing.cfg"; #inpWidth = 320 , inpHeight = 320 
# modelWeights = "yolov3_training_last.weights"; 


modelConfiguration = "yolov4-custom.cfg"; #inpWidth = 320 , inpHeight = 320 
modelWeights = "yolov4-custom_last.weights"; 
print("[INFO] loading model...")
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

writer = None
 
W = None
H = None 


ct = CentroidTracker(maxDisappeared=5, maxDistance=100)  #10 60  #5 100
trackers = []
trackableObjects = {}

global ay_count
ay_count = []


totalDown = 0
totalUp = 0
totalAll = 0


def getOutputsNames(net):
    layer_names = net.getLayerNames()
    # print(layer_names)
    # return [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()] ############################################  cpu
    return [layer_names[i-1] for i in net.getUnconnectedOutLayers()] ############################################  gpu

def drawPred(classId, conf, left, top, right, bottom):
    # วาดกรอบคน
    if classId == 0:
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 2)
    # # cv2.putText(frame, classId, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    # else:
    #     cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 255), 2)

    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

def postprocess(frame, outs):
    global ay_count,classIds,box
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    rects = []
    classIds = []
    confidences = []
    boxes = []
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


    for i in indices:
        # i = i[0] ###########################################  cpu
        i = i ###########################################  gpu
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        # Class "person"
        if classIds[i] == 0 or classIds[i] == 1:
            rects.append((left, top, left + width, top + height))
            objects = ct.update(rects)
            counting(objects)
            drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
            # print("นี้คือ",classIds[i])
            # if classIds[i] == 0:
            #     print("mask")

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
            # print(objectID," : ",direction)

            if not to.counted:
                if direction < 0 and centroid[1] in range(frameHeight//2 - 100, frameHeight//2 + 100):
                    totalUp += 1
                    # totalAll += 1
                    dt = datetime.now()
                    date = str(dt.year)+"-"+ str(dt.month)+"-"+ str(dt.day)
                    time = str(dt.hour)+":"+ str(dt.minute)+":"+ str(dt.second)
                    ay_count.append([totalAll,date,time,'Out'])
                    to.counted = True
                    # print("OUT  :   ",totalUp)
                    # print(ay_count)

                    # Tablae_People.insert(parent='',index='end',iid=sum_people,text='',
                    # values=(sum_people,date,time,'Out'))
                    # Tablae_People.see(sum_people)
                    # sum_people = sum_people + 1
                    if classIds[i] == 0:
                        print("mask")
                        Tablae_People.insert(parent='',index='end',iid=sum_people,text='',
                        values=(sum_people,date,time,'Out','mask'))
                        Tablae_People.see(sum_people)
                        sum_people = sum_people + 1
                        # img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
                        # img_temp = img_rgb[int(box[1]):int(box[1]) + int(box[3]), int(box[0]):int(box[0]) + int(box[2]), :]
                        # curr_datetime = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
                        # img_name = "{}{}.jpg".format("nomask",curr_datetime)
                        # cv2.imwrite(os.path.join(pathnomask,img_name),img_temp)
                        # with con:
                        #     # now = datetime.today()
                        #     # dtwithoutseconds = now.replace(second=0, microsecond=0)
                        #     name = "unknow"
                        #     cur = con.cursor()
                        #     sql = "insert into tb_member (usr_name,usr_date,usr_time,usr_status,usr_statusmask,usr_pic) VALUES (%s,%s,%s,%s,%s,%s)"
                        #     cur.execute(sql,(name,date,time,"out","mask",pathnomask+"nomask"+curr_datetime+".jpg"))
                        #     con.commit()
                    if classIds[i] == 1:
                        print("Nomask")
                        Tablae_People.insert(parent='',index='end',iid=sum_people,text='',
                        values=(sum_people,date,time,'Out','Nomask'))
                        Tablae_People.see(sum_people)
                        sum_people = sum_people + 1
                        img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
                        img_temp = img_rgb[int(box[1]):int(box[1]) + int(box[3]), int(box[0]):int(box[0]) + int(box[2]), :]
                        curr_datetime = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
                        img_name = "{}{}.jpg".format("nomask",curr_datetime)
                        cv2.imwrite(os.path.join(pathnomask,img_name),img_temp)
                        with con:
                            # now = datetime.today()
                            # dtwithoutseconds = now.replace(second=0, microsecond=0)
                            name = "unknow"
                            cur = con.cursor()
                            sql = "insert into tb_member (usr_name,usr_date,usr_time,usr_status,usr_statusmask,usr_pic) VALUES (%s,%s,%s,%s,%s,%s)"
                            cur.execute(sql,(name,date,time,"out","mask",pathnomask+"nomask"+curr_datetime+".jpg"))
                            con.commit()
                        
                        
                        
                        # img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
                        # img_temp = img_rgb[int(box[1]):int(box[1]) + int(box[3]), int(box[0]):int(box[0]) + int(box[2]), :]
                        # curr_datetime = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
                        # img_name = "{}{}.jpg".format("nomask",curr_datetime)
                        # cv2.imwrite(os.path.join(pathnomask,img_name),img_temp)
                            
                        

                elif direction > 0 and centroid[1] in range(frameHeight//2 - 100, frameHeight//2 + 100):
                    totalDown += 1
                    # totalAll += 1
                    dt = datetime.now()
                    date = str(dt.year)+"-"+ str(dt.month)+"-"+ str(dt.day)
                    time = str(dt.hour)+":"+ str(dt.minute)+":"+ str(dt.second)
                    ay_count.append([totalAll,date,time,'In'])
                    to.counted = True
                    # print("IN  :   ",totalDown)
                    # print(ay_count)
                    if classIds[i] == 0:
                        print("mask")
                        Tablae_People.insert(parent='',index='end',iid=sum_people,text='',
                        values=(sum_people,date,time,'in','mask'))
                        Tablae_People.see(sum_people)
                        sum_people = sum_people + 1
                    if classIds[i] == 1:
                        print("Nomask")
                        Tablae_People.insert(parent='',index='end',iid=sum_people,text='',
                        values=(sum_people,date,time,'in','Nomask'))
                        Tablae_People.see(sum_people)
                        sum_people = sum_people + 1
                
        
        trackableObjects[objectID] = to
    
        # #แสดงตัวเลขคน และ จุดของคน
        text = "ID {}".format(objectID)
        # if(dotpeople == "open"):
        #     cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        # if(IDpeople == "open"):
        #     cv2.putText(frame, text, (centroid[0] - 15, centroid[1] - 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (9, 207, 255), 2) #BGR
    
def RunToMain():
    root.deiconify()
    win.destroy()

# ฟังก์ชั่นปุ่มกดออกจาก GUI
def ExitApplication():
    MsgBox = tk.messagebox.askquestion ('Exit Application','คุณต้องการจะจบการทำงานหรือไม่',icon = 'warning')
    if MsgBox == 'yes':
       root.destroy()


# ฟังชั่นนับเข้าออก
def run_detect():

    global frame,image,fps_end_time,fps_start_time,count_frame

    count_frame = count_frame + 1
    # print(count_frame)
    ###### FPS ######
    fps_end_time = time.time()
    time_diff = fps_end_time - fps_start_time
    fps = 1/(time_diff)
    fps_start_time = fps_end_time
    fps_text = "FPS: {: .2f}".format(fps)

    # cv2image1= cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # img1 = Image.fromarray(cv2image1)
    # imgtk1 = ImageTk.PhotoImage(image = img1)
    # label01.imgtk = imgtk1
    # label01.configure(image=imgtk1)
    
    ##!!YOLO!!
    #### People Detect ######
    hasFrame, frame = webcam.read()
    frame = cv2.resize(frame, (640,480), interpolation = cv2.INTER_AREA)
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

        # cv2.line(frame, (0,frameHeight//2 - 50), (frameWidth,frameHeight//2 + 50), (0, 0, 255), 2) 
    cv2.rectangle(frame, (0,frameHeight//2 - 100), (frameWidth,frameHeight//2 + 100), (0, 0, 255), 2) 
    
    cv2.line(frame, (0, frameHeight // 2), (frameWidth, frameHeight // 2), (0, 255, 255), 2)


    info = [("Out", totalUp),("In", totalDown),]
    for (i, (k, v)) in enumerate(info):
        text = "{} : {}".format(k, v)
        cv2.putText(frame, text, (10, 50 - ((i * 20))),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 230), 2)

        blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

        net.setInput(blob)
    if(count_frame > 0):
        outs = net.forward(getOutputsNames(net))
        postprocess(frame, outs)
        count_frame = 0

    cv2.putText(frame, fps_text, (10, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),2)        
    #!!yolo
    # t, _ = net.getPerfProfile()
    # label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    # cv2.putText(frame, label, (10, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    cv2image2= cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    img2 = Image.fromarray(cv2image2)
    imgtk2 = ImageTk.PhotoImage(image = img2)
    label02.imgtk = imgtk2
    label02.configure(image=imgtk2)


    win.after(10, run_detect)
# เรียกแสดงเว็ป
def link():
    webbrowser.open_new("http://localhost/skydash-free-bootstrap-admin-template-main/template/index.php")
    # with con:
    #     cur = con.cursor()
    #     sql = "SELECT * FROM tb_member"
    #     cur.execute(sql)
    #     rows = cur.fetchall()
    datatext = []
        # for mem_id in rows:
    print(datatext)
    # print('This is standard output', file=sys.stdout)
    # print('Hello world!', file=sys.stderr)
        # print('This is standard output', file=sys.stdout)
    # known_face_names, known_face_encodings = pickle.load(open('faces.p','rb'))
    path = glob.glob("C:/Users/User/Desktop/project/mainproject/test/*.jpg")
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
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        for face_encoding , face_location in zip(face_encodings, face_locations):
            face_distances = face_recognition.face_distance(known_face_encodings,face_encoding)
            best_match_index = np.argmin(face_distances)
            # print(face_distances)
            # print(best_match_index)
            if (face_distances < 0.8).all():
                    name = known_face_names[best_match_index]
                    top, right, bottom, left = face_location
                    draw.rectangle([left,top,right,bottom])
                    draw.text((left,top), name)
            
            else:
                name = "unknow"
                top, right, bottom, left = face_location
                draw.rectangle([left,top,right,bottom])
                draw.text((left,top), name)
            datatext.append(name)
            print(datatext.append(name))
           
            with con:
                cur = con.cursor()
                # sql = "INSERT INTO tb_memberallow (mem_fname) VALUES (%s)"
                sql = "SELECT * FROM tb_member"
                cur.execute(sql)
                rows = cur.fetchall()
         
         
    for usr_id in range(len(datatext)):
                    # datatext.append(name)
                        # print(mem_id,name,count)
        sql = "UPDATE tb_member SET usr_name = (%s) WHERE usr_id = (%s)"
        cur.execute(sql,(datatext[usr_id],usr_id+1))
        con.commit()

    
# เรียกแสดงหน้า detec
def run():
    root.withdraw()
    global ay_count,Tablae_Hand,Hand_DataTable,Tablae_People
    global win,label01,label02
    win = Toplevel(root)
    win.geometry("1100x781")#655x590 #1305x590
    win.resizable(width=False, height=False)
    # label01 =Label(win)
    # label01.place(x = 5, y = 80)
    # vertical =Frame(win, bg='#C2C2C2', height=645,width=2)
    # vertical.place(x=651, y=80)
    # horizontal =Frame(win, bg='#C2C2C2', height=2,width=1290)
    # horizontal.place(x=7, y=723)
    label04 = Label(win, text = 'People Detect', font=('calibre 20 underline bold')).pack()
    label02 =Label(win)
    label02.pack()
    # label03 = Label(win, text = 'Hand Detect', font=('Helvetica 20 underline bold')).place(x = 5, y = 8)
    
    btn001 = Button(win,text="ย้อนกลับ", fg="white",command=RunToMain, font=('calibre',16), bg="#369F36", width=10)
    # btn001.place(x = 7, y = 730)
    # btn001 = Button(win,text="ดูการการวิเคราะห์และรายละเอียด", fg="white",command=link, font=('calibre',16), bg="#369F36", width=20)
    btn001.place(x = 50, y = 730)
    btn003 = Button(win,text="จบการทำงาน", fg="white",command=ExitApplication, font=('calibre',16), bg="#B9062F", width=10)
    btn003.place(x = 900, y = 730)#517x540

    # run_hand()
    # run_people()
    run_detect()

    ####### People Table ######
    Tablae_People_scroll = Scrollbar(win)
    Tablae_People_scroll.place(x = 1280, y = 570 , height=150)
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

# ส่วนด้านบน gui
l1 = Label(root, text = '', font=('calibre',5, 'bold')).pack()
l1 = Label(root, text = 'TEST', font=('calibre',16, 'bold')).pack()
l1 = Label(root, text = '"TEST"', font=('calibre',16, 'bold')).pack()
l1 = Label(root, text = '', font=('calibre',2, 'bold')).pack()
line01 =Frame(root, bg='#C2C2C2', height=2,width=265).pack(padx = 5, pady = 10)
# 
# ปุ่ม gui
btn03 = Button(root,text="Run", fg="white", font=('calibre',16, 'bold'), bg="blue", command=run, width=20).pack(padx = 5, pady = 5)#.grid(row=1,column=0).pack()
btn03 = Button(root,text="Exit", fg="white", font=('calibre',16, 'bold'), bg="#B9062F", command=ExitApplication, width=20).pack(padx = 5, pady = 5)#.grid(row=1,column=0).pack()
btn01 = Button(root,text="แสดงรายละเอียด", fg="white", font=('calibre',16, 'bold'), bg="#00FF00",command=link, width=20).pack(padx = 5, pady = 5)
# 
if __name__ == '__main__':

    try:
        root.mainloop()
        
    except SystemExit:
        pass