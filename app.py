import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Load pre-trained models for face, age, and gender detection
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

# Initialize models
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Helper function to detect faces
def detect_faces(net, frame, conf_threshold=0.7):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    face_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame.shape[1])
            y1 = int(detections[0, 0, i, 4] * frame.shape[0])
            x2 = int(detections[0, 0, i, 5] * frame.shape[1])
            y2 = int(detections[0, 0, i, 6] * frame.shape[0])
            face_boxes.append([x1, y1, x2, y2])
    return face_boxes

# Helper function to get age and gender predictions
def predict_age_gender(faceNet, ageNet, genderNet, face_img):
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    genderNet.setInput(blob)
    gender_preds = genderNet.forward()
    gender = genderList[gender_preds[0].argmax()]

    ageNet.setInput(blob)
    age_preds = ageNet.forward()
    age = ageList[age_preds[0].argmax()]
    
    return gender, age

# Streamlit Interface
st.title("Age and Gender Prediction")
st.write("This application detects age and gender from a live camera or an uploaded image.")

# Option to choose camera or image upload
option = st.radio("Choose an input method:", ("Camera", "Upload Image"))

if option == "Camera":
    # Start the webcam feed if 'Open Camera' is clicked
    if st.button("Open Camera", key="open_camera"):
        st.write("Click 'Stop' to end the camera.")
        video_capture = cv2.VideoCapture(0)
        while True:
            _, frame = video_capture.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_boxes = detect_faces(faceNet, frame)
            for box in face_boxes:
                x1, y1, x2, y2 = box
                face = frame[y1:y2, x1:x2]
                gender, age = predict_age_gender(faceNet, ageNet, genderNet, face)
                label = f"{gender}, {age}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            st.image(frame, channels="RGB")
            if st.button("Stop", key="stop_camera"):
                video_capture.release()
                break

elif option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        face_boxes = detect_faces(faceNet, frame)
        for box in face_boxes:
            x1, y1, x2, y2 = box
            face = frame[y1:y2, x1:x2]
            gender, age = predict_age_gender(faceNet, ageNet, genderNet, face)
            label = f"{gender}, {age}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        st.image(frame, caption='Processed Image', use_column_width=True)
