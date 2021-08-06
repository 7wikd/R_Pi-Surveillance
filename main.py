import os
import cv2
import time
import face_recognition
import pickle

pTime = 0
cTime = 0

KNOWN_FACES_DIR = 'known_faces'
UNKNOWN_FACES_DIR = 'unknown_faces'
TOLERANCE = 0.7
FRAME_THICKNESS = 2
FONT_THICKNESS = 2
MODEL = 'cnn'

cap = cv2.VideoCapture(0)

known_faces = []
known_names = []

if not os.path.exists(f"./{KNOWN_FACES_DIR}"):
    os.mkdir(KNOWN_FACES_DIR)

for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        encoding = pickle.load(open(f"{name}/{filename}","rb"))
        
        known_faces.append(encoding)
        known_names.append(int(name))

if len(known_names) > 0:
    next_id = max(known_names)+1
else:
    next_id = 0


while True:
    success, image = cap.read()

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    locations = face_recognition.face_locations(image,model=MODEL)
    encodings = face_recognition.face_encodings(image,locations)
    
    for face_encoding, face_location in zip(encodings,locations):
        results = face_recognition.compare_faces(known_faces,face_encoding,TOLERANCE)
        match = None

        if True in results:
            match = known_names[results.index(True)]
            print(f"Match Found:{match}")

        else:
            match = str(next_id)
            next_id += 1
            
        top_left = (face_location[3],face_location[0])
        bottom_right = (face_location[1],face_location[2])

        color = [0,255,0]
        cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

        top_left = (face_location[3],face_location[2])
        bottom_right = (face_location[1],face_location[2]+22)
        cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
        cv2.putText(image,match, (face_location[3]+10,face_location[2]+15),cv2.FONT_HERSHEY_SIMPLEX,0.5, (200,200,0))


    cv2.putText(image,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,2,(255,255,0),3)
    cv2.imshow(" ",image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break