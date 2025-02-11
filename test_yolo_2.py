import argparse
import os

import cv2
import torch


def main():
    # Configuration des arguments
    parser = argparse.ArgumentParser(description="Analyse vidéo avec YOLO.")
    parser.add_argument("-v", "--video", type=str, help="Chemin de la vidéo source. Utilise la caméra si non spécifié.")
    parser.add_argument("-o", "--output", type=str, help="Chemin du fichier de sortie. Affiche uniquement en direct si non spécifié.")
    parser.add_argument("-c", "--camera", type=int, default=0, help="Index de la caméra à utiliser si aucune vidéo n'est fournie.")

    args = parser.parse_args()

    # Charger le modèle YOLOv5 pré-entraîné
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Déterminer la source de la vidéo
    if args.video:
        if not os.path.exists(args.video):
            print(f"Erreur : le fichier vidéo '{args.video}' n'existe pas.")
            return
        cap = cv2.VideoCapture(args.video)
    else:
        print("Aucun chemin vidéo fourni. Utilisation de la caméra.")
       # cap = cv2.VideoCapture(0)
        cap = cv2.VideoCapture(args.camera)
    # Configurer la sortie vidéo si nécessaire
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec vidéo
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Défaut à 30 FPS si non disponible
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # Analyser la vidéo frame par frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Fin de la vidéo ou erreur de lecture.")
            break

        # Faire une détection d'objets
        results = model(frame)

        # Annoter la frame avec les résultats
        annotated_frame = results.render()[0]

        # Enregistrer la frame annotée si un fichier de sortie est spécifié
        if out:
            out.write(annotated_frame)

        # Afficher la vidéo en direct
        cv2.imshow('YOLO Detection', annotated_frame)

        # Quitter avec la touche 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Arrêt demandé par l'utilisateur.")
            break

    # Libérer les ressources
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

