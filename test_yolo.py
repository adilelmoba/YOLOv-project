"""
Ce script analyse une vidéo en utilisant le modèle YOLOv5.
Il utilise PyTorch et OpenCV pour la détection en temps réel.
"""
import warnings

import cv2
import torch

warnings.filterwarnings('ignore', category=FutureWarning)


# Charger le modèle YOLOv5 pré-entraîné
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Ouvrir la vidéo
video_path = 'ressources/VisionComputing_Charion_min5.mp4'
cap = cv2.VideoCapture(video_path)

# Lire et analyser chaque image de la vidéo
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Faire une détection d'objet
    results = model(frame)

    # Afficher les résultats sur l'image
    annotated_frame = results.render()[0]

    # Afficher la vidéo en direct
    cv2.imshow('YOLO Detection', annotated_frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()

