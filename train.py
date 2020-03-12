import tkinter as tk
from tkinter import Message, Text
import cv2, os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font

window = tk.Tk()
window.title("Facial Recognition Attendance System")

dialog_title = 'QUIT'
dialog_text = 'Are you sure?'
# answer = messagebox.askquestion(dialog_title, dialog_text)

window.geometry('850x600')

#window.configure(background='white')

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

lbl = tk.Label(window, text="Enter ID", width=20, height=2, font=('times', 15, ' bold '))
lbl.place(x=55, y=100)

txt = tk.Entry(window, width=20, font=('times', 15, ' bold '))
txt.place(x=375, y=115)

lbl2 = tk.Label(window, text="Enter Name", width=20, height=2, font=('times', 15, ' bold '))
lbl2.place(x=55, y=200)

txt2 = tk.Entry(window, width=20, font=('times', 15, ' bold '))
txt2.place(x=375, y=215)

lbl3 = tk.Label(window, text="Notification : ", width=20, height=2,
                font=('times', 15, ' bold underline '))
lbl3.place(x=55, y=300)

message = tk.Label(window, text="", bg="white", fg="black", width=30, height=2, activebackground="red",
                   font=('times', 15, ' bold '))
message.place(x=375, y=300)

lbl3 = tk.Label(window, text="Attendance : ", width=20, height=2,
                font=('times', 15, ' bold  underline'))
lbl3.place(x=55, y=500)

message2 = tk.Label(window, text="", fg="white", bg="white", activeforeground="green", width=30, height=2,
                    font=('times', 15, ' bold '))
message2.place(x=410, y=500)


n_train=0;
n_start=0;

def clear():
    txt.delete(0, 'end')
    res = ""
    message.configure(text=res)


def clear2():
    txt2.delete(0, 'end')
    res = ""
    message.configure(text=res)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def TakeTrainImages():

    
    window.withdraw()
    cam = cv2.VideoCapture(0)

    
    Id = (txt.get())
    name = (txt2.get())
    if Id == "":
    
        res="Enter Numeric Id"
        message.configure(text=res)
    if (is_number(Id) and name.isalpha()):
        
        
        
        ret, im = cam.read()
       
        w_frame=im.shape[1]
        h_frame=im.shape[0]
        
        rx=int(w_frame*0.35)
        ry=int(h_frame*0.2)
        rw=int(w_frame*0.3)
        rh=int(h_frame*0.3)
    
        
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0
        
        while (True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.rectangle(img, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
            
            faces = detector.detectMultiScale(gray[ry:ry+rh, rx:rx+rw], 1.3, 5)

                             
            
            for (x, y, w, h) in faces:
                if w>0.6*rw:
                    x=x+rx
                    y=y+ry
                
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    # incrementing sample number
                    sampleNum = sampleNum + 1
                    # saving the captured face in the dataset folder TrainingImage
                    cv2.imwrite("TrainingImage\ " + name + "." + Id + '.' + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
                    # display the frame
            cv2.imshow('frame', img)        
            # wait for 100 miliseconds
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 100
            elif sampleNum > 60:
                window.deiconify()
                break
       
        res = "Images Saved for ID : " + Id + " Name : " + name
        row = [Id, name]
        with open('StudentDetails\StudentDetails.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text=res)
    else:
        if (is_number(Id)):
            res = "Enter Alphabetical Name"
            message.configure(text=res)
        if (name.isalpha()):
            res = "Enter Numeric Id"
            message.configure(text=res)
    cam.release()
    cv2.destroyAllWindows()    
    recognizer = cv2.face_LBPHFaceRecognizer.create()  # recognizer = cv2.face.LBPHFaceRecognizer_create()#$cv2.createLBPHFaceRecognizer()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel\Trainner.yml")
    res = "Image Trained"  # +",".join(str(f) for f in Id)
    message.configure(text=res)    


   
    


def getImagesAndLabels(path):
    # get the path of all the files in the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # print(imagePaths)

    # create empth face list
    faces = []
    # create empty ID list
    Ids = []
    # now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        # loading the image and converting it to gray scale
        pilImage = Image.open(imagePath).convert('L')
        # Now we are converting the PIL image into numpy array
        imageNp = np.array(pilImage, 'uint8')
        # getting the Id from the image
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids


    
def TakeAttendence():
    global n_train, n_start
    n_train=1
    n_start=1
  
    window.withdraw()
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
    recognizer.read("/Users/user/Downloads/fyp-master5/TrainingImageLabel/Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);
    df = pd.read_csv("/Users/user/Downloads/fyp-master5/StudentDetails/StudentDetails.csv")
   
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)
    nn=0;
    n_unknown=0
    cam = cv2.VideoCapture(0)
    
    #cv2.namedWindow("frame",cv2.WND_PROP_FULLSCREEN)
    
    cv2.namedWindow("frame")
    
   
    #cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    ret, im = cam.read()
       
    w_frame=im.shape[1]
    h_frame=im.shape[0]
    
    rx=int(w_frame*0.35)
    ry=int(h_frame*0.2)
    rw=int(w_frame*0.3)
    rh=int(h_frame*0.3)
    
   
    if rw>rh:
        rh=rw
    else:
        rw=rh       
      
    prevId=0
    n_disp=20
    disp_x=0
    disp_y=0
    while True:
        
        

        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        
        cv2.rectangle(im, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
        faces = faceCascade.detectMultiScale(gray[ry:ry+rh, rx:rx+rw], 1.2, 5)
               
        
        for (x, y, w, h) in faces:
            if w>0.6*rw:
                x=x+rx
                y=y+ry
                
           
                cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
                Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
               
                if (conf < 50):
                    nn=nn+1
                    ts = time.time()
                    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                    aa = df.loc[df['Id'] == Id]['Name'].values
                             
                    tt = str(Id) + "-" + aa
                    attendance.loc[len(attendance)] = [Id, aa, date, timeStamp]
                    
    
                else:
                    
                    Id = ''
                    tt = str(Id)
                if (conf > 75):
                    Id = 'Unknown'
                    tt = str(Id)
                    n_unknown=n_unknown+1
                    noOfFile = len(os.listdir("ImagesUnknown")) + 1
                    cv2.imwrite("ImagesUnknown\Image" + str(noOfFile) + ".jpg", im[y:y + h, x:x + w])
                cv2.putText(im, str(tt), (x, y + h), font, 1, (255, 255, 255), 2)
                
        if nn>5:
            nn=0;
            if prevId != Id:
                attendance.loc[len(attendance)] = [Id, aa, date, timeStamp]
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                Hour, Minute, Second = timeStamp.split(":")
                fileName = "Attendance\Attendance_" + date + ".csv"
                attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
                attendance.to_csv(fileName, index=False)
                attendance.to_csv()
                prevId=Id
                
                n_disp=0
                disp_x=x
                disp_y=y
                cv2.putText(im, "Attendance confirmed", (disp_x-int(rw/3), disp_y-int(rh/4)), font, 1, (255, 255, 255), 2)
        if n_disp<20:
            n_disp=n_disp+1
            cv2.putText(im, "Attendance confirmed", (disp_x-int(rw/3), disp_y-int(rh/4)), font, 1, (255, 255, 255), 2)
                
        if n_unknown>10:
            
            notics = tk.Tk()

            notics.geometry('200x60')

            notics.grid_rowconfigure(0, weight=1)
            notics.grid_columnconfigure(0, weight=1)


            lbl1 = tk.Label(notics, text="You must register", width=15, height=2,
                font=('times', 15 ))
            lbl1.place(x=0, y=5)

       
            notics.update_idletasks()
            notics.update()
            cv2.waitKey(1000)
            notics.destroy()
            
            n_train=1
            
            window.deiconify()
            
            break
            
            
        if (cv2.waitKey(1) == ord('q')):
            break
         
        #attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
        cv2.imshow('frame', im)
        
        
   
    cam.release()
    cv2.destroyAllWindows()
    res = attendance
    message2.configure(text=res)
  

clearButton = tk.Button(window, text="Clear", command=clear, fg="white", bg="red", width=10, height=1,
                        activebackground="Red", font=('times', 15, ' bold '))
clearButton.place(x=650, y=107.5)
clearButton2 = tk.Button(window, text="Clear", command=clear2, fg="white", bg="red", width=10, height=1,
                         activebackground="Red", font=('times', 15, ' bold '))
clearButton2.place(x=650, y=207.5)
takeImg = tk.Button(window, text="Enrol", command=TakeTrainImages, fg="white", bg="red", width=10, height=1,
                    activebackground="Red", font=('times', 15, ' bold '))
takeImg.place(x=125, y=400)

trackImg = tk.Button(window, text="Take Attendence", command=TakeAttendence, fg="white", bg="red", width=12, height=1,
                     activebackground="Red",font=('times', 15, ' bold '))

trackImg.place(x=405, y=400)

quitWindow = tk.Button(window, text="Quit", command=window.destroy, fg="white", bg="red", width=5, height=1,
                       activebackground="Red", font=('times', 15, ' bold '))
quitWindow.place(x=685, y=400)


#window.mainloop()
#window.update_idletasks()
#window.update()

while 1:
    
    if n_start==1:
        window.update_idletasks()
        window.update()
    if n_train==0:
        TakeAttendence()
    
  
