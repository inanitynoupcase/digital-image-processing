import dlib
from argparse import ArgumentParser
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import face_recognition
from utils import (
    load_image,
    face_image_to_encoding
)


img = load_image('images/team_noise.png')

#FILTER
#Gauss 
#blur = cv.GaussianBlur(img,(5,5),0)
#Lọc trung vị
blur = cv.medianBlur(img,5)
#Bilateral blur = cv.bilateralFilter(img,9,75,75)

#EDGES DETECTION
#canny
edges = cv.Canny(blur, 100, 200)

# nhận diện khuôn mặt trong ảnh và phát hiện khuôn mặt

# trích xuất face embeddings / encodings của ảnh khuôn mặt đã biết
green_color = (0, 255, 0)
known_faces = [
    ("cong dang", "images/cong_dang_037.jpg"),
    ("gia bao", "images/gia_bao_017.jpg"),
    ("duc thang", "images/duc_thang_193.jpg"),
    ("cao ky", "images/cao_ky_084.jpg"),   
    ("viet cuong", "images/viet_cuong_031.jpg"),
    ("tuan khoi", "images/tuan_khoi_094.jpg")
]

known_face_names = []
known_face_encodings = []
for name, image_path in known_faces:
    #image = cv.imread(image_path)
    image = load_image(image_path)
    encoding = face_image_to_encoding(image)
    known_face_names.append(name)
    known_face_encodings.append(encoding)

# Load an image with an unknown face
unknown_image = blur.copy()
# Find all the faces and face encodings in the unknown image
face_locations = face_recognition.face_locations(unknown_image, number_of_times_to_upsample=0, model='hog')
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
print(len(face_locations))
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown"
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    #print(face_distances)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]
    #ve hinh dua chu
    cv.rectangle(unknown_image, pt1=(left, top), pt2=(right, bottom), color=green_color, thickness=2)
    cv.putText(unknown_image, name, (left, top), cv.FONT_HERSHEY_SIMPLEX, 1.1, (36,255,12), 2)#code dua chu

# Display
plt.figure(figsize=(20,30))
#figure(num=None,figsize=(100,100),dpi=80,facecolor='w',edgecolor='k')
plt.subplot(221),plt.imshow(img),plt.title('Noise')
plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(blur),plt.title('Noise - Blurred')
plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(edges, cmap='gray'),plt.title('Edges')
plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(unknown_image),plt.title('Faces')
plt.xticks([]), plt.yticks([])
plt.show()



