from re import X
from select import select
from flask import Blueprint,render_template,request,redirect,url_for,session
# import pymysql
import pymysql
# from config import *
from config import *
import os
import sys

import glob
from PIL import Image, ImageDraw
import face_recognition
import pickle
import numpy as np

member = Blueprint('member',__name__)
con = pymysql.connect(HOST,USER,PASS,DATABASE)
@member.route("/createreport")
def Createreport():
    if "username" not in session:
        return render_template('login.html',headername="Login เข้าใช้งานระบบ")
    with con:
        cur = con.cursor()
        sql = "SELECT * FROM tb_memberallow2"
        cur.execute(sql)
        rows = cur.fetchall()
        datatext = []
        # for mem_id in rows:
        #     print(mem_id)
    # print('This is standard output', file=sys.stdout)
    # print('Hello world!', file=sys.stderr)
        # print('This is standard output', file=sys.stdout)
    # known_face_names, known_face_encodings = pickle.load(open('faces.p','rb'))
    path = glob.glob("C:/Users/asus/Desktop/project/flask-facemaskdetection-main/static/testpeople/*.jpg")
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
            if (face_distances < 0.8).any():
                    name = known_face_names[best_match_index]
                    top, right, bottom, left = face_location
                    draw.rectangle([left,top,right,bottom])
                    draw.text((left,top), name)
                    datatext.append(name)
            
            else:
                name = "unknow"
                top, right, bottom, left = face_location
                draw.rectangle([left,top,right,bottom])
                draw.text((left,top), name)
                datatext.append(name)
                
           
            with con:
                cur = con.cursor()
                # sql = "INSERT INTO tb_memberallow (mem_fname) VALUES (%s)"
                sql = "SELECT * FROM tb_memberallow2"
                cur.execute(sql)
                rows = cur.fetchall()
         
    # print (datatext,file=sys.stdout)
         
    for usr_id in range(len(datatext)):
                    # datatext.append(name)
                        # print(mem_id,name,count)
        sql = "UPDATE tb_member2 SET usr_name = (%s) WHERE usr_id = (%s)"
        cur.execute(sql,(datatext[usr_id],mem_id+1))
        con.commit()
            
                    # ต้องทำloop เพื่อนำตัวแปรcount มา+เพิ่มเพื่อใส่ใน mem_id ตอนนี้ยังเอา array ตัวสุดท้ายไปใส่ในmem_fnameอยู่XXXX
        # print (datatext[mem_id],file=sys.stdout)
    # print (datatext,file=sys.stdout)
            # con.commit()
            

            
                # print(face_distances)
                # print(name)
        # print(name)
        # image.show()
    return render_template("report.html",headername="ข้อมูลสมาชิก",datas=rows)