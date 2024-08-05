import face_recognition
import cv2
import numpy as np
import csv
import os
import glob
from datetime import datetime

now = datetime.now()
year = now.strftime('%Y')
month = now.strftime('%m')
date = now.strftime('%d-%m-%Y')
time = [now.hour, now.minute]

try:
    os.mkdir('photos')
except:
    pass
try:
    os.mkdir('attendance_database')
except:
    pass
try:
    os.mkdir('attendance_database/'+year)
except:
    pass
try:
    os.mkdir('attendance_database/'+year+'/'+month)
except:
    pass
try:
    os.mkdir('attendance_database/'+year+'/'+month+'/'+date)
except:
    pass

video_capture = cv2.VideoCapture(0)
image_names = glob.glob('photos/*')
images = [face_recognition.load_image_file(i) for i in image_names]
known_face_encoding = [face_recognition.face_encodings(i)[0] for i in images]

known_faces_names = [(i.split('/')[1]).split('.')[0] for i in image_names]

students = known_faces_names.copy()

face_locations = []
face_encodings = []
face_names = []

if now < now.replace(hour=10, minute=30, second=00):
    period = 'Extra class (Pre-routine)'
elif now < now.replace(hour=11, minute=20, second=00):
    period = '1st period'
elif now < now.replace(hour=12, minute=10, second=00):
    period = '2nd period'
elif now < now.replace(hour=1, minute=00, second=00):
    period = '3rd period'
elif now < now.replace(hour=1, minute=40, second=00):
    period = 'Extra class (During lunch break)'
elif now < now.replace(hour=2, minute=30, second=00):
    period = '4th period'
elif now < now.replace(hour=3, minute=20, second=00):
    period = '5th period'
elif now < now.replace(hour=4, minute=10, second=00):
    period = '6th period'
else:
    period = 'Extra class (Post-routine)'

f = open('attendance_database/'+year+'/'+month+'/'+date+'/'+period+'.csv','w+',newline = '')
lnwriter = csv.writer(f)

while True:
    _,frame = video_capture.read()
    rgb_small_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encoding,face_encoding, tolerance = 0.5)
        name=""
        face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
        best_match_index = np.argmin(face_distance)
        if matches[best_match_index]:
            name = known_faces_names[best_match_index]

        face_names.append(name)
        if name in known_faces_names:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10,100)
            fontScale              = 1
            fontColor              = (255,0,0)
            thickness              = 3
            lineType               = 2

            cv2.putText(frame,name+' Present', bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)

            if name in students:
                print(name)
                students.remove(name)
                # print(students)
                current_time = now.strftime("%H-%M-%S")
                lnwriter.writerow([name,current_time])
    cv2.imshow("attendence system",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
video_capture.release()
cv2.destroyAllWindows()
f.close()
