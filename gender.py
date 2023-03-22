import cv2

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
scale=1/255.0
def load_models():
    age_net= cv2.dnn.readNetFromCaffe('models/deploy_age.prototxt','models/age_net.caffemodel')
    gender_net= cv2.dnn.readNetFromCaffe('models/deploy_gender.prototxt','models/gender_net.caffemodel')
    return (age_net,gender_net)

def predict_gender(faces):
    blob=cv2.dnn.blobFromImage(faces,scale,(227,227),MODEL_MEAN_VALUES,swapRB=False,crop=False)
    gender_net.setInput(blob)
    return gender_net.forward()

def predict_age(faces):
    blob=cv2.dnn.blobFromImage(faces,scale,(227,227),MODEL_MEAN_VALUES,swapRB=False,crop=False)
    age_net.setInput(blob)
    return age_net.forward()

def play_video(age_net,gender_net,cap):
    while True:
        ret,frame=cap.read()
        color=cv2.cvtColor(frame,cv2.COLOR_BGR2RGBA)
        face_cascade=cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
        faces=face_cascade.detectMultiScale(color,1.3,5)
        # print(faces)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
            img=frame[x:x+h,y:y+w].copy()
            gender_pred=predict_gender(img)
            # print(gender_pred)
            gender="Gender: "+gender_list[gender_pred[0].argmax()]
            # print(gender)
            age_pred=predict_age(img)
            age="Age : "+age_list[age_pred[0].argmax()]
            cv2.putText(frame,gender,(x+w,y-30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(120,250,0),1)
            cv2.putText(frame,age,(x+w,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(120,250,0),1)
            

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    age_net,gender_net=load_models()
    cap=cv2.VideoCapture(0)
    gender_list=['Male','Female']
    age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
    play_video(age_net,gender_net,cap)
