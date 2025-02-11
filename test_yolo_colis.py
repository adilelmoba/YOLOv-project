import argparse
import os
import cv2
import torch

# Définition des critères de validation des colis
MIN_WIDTH, MAX_WIDTH = 50, 300  # Largeur min et max en pixels
MIN_HEIGHT, MAX_HEIGHT = 50, 300  # Hauteur min et max en pixels
VALID_CLASSES = ["box", "package", "parcel"]  # Classes considérées comme colis

def main():
    # Configuration des arguments CLI
    parser = argparse.ArgumentParser(description="Analyse vidéo avec YOLOv5s.")
    parser.add_argument("-v", "--video", type=str, help="Chemin de la vidéo source. Utilise la caméra si non spécifié.")
    parser.add_argument("-o", "--output", type=str, help="Chemin du fichier de sortie. Affiche uniquement en direct si non spécifié.")
    parser.add_argument("-c", "--camera", type=int, default=0, help="Index de la caméra à utiliser si aucune vidéo n'est fournie.")

    args = parser.parse_args()

    # Charger le modèle YOLOv5s depuis Torch Hub
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Déterminer la source vidéo
    if args.video:
        if not os.path.exists(args.video):
            print(f"Erreur : le fichier vidéo '{args.video}' n'existe pas.")
            return
        cap = cv2.VideoCapture(args.video)
    else:
        print("Aucun chemin vidéo fourni. Utilisation de la caméra.")
        cap = cv2.VideoCapture(args.camera)

    # Configurer la sortie vidéo si nécessaire
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # Analyser la vidéo frame par frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Fin de la vidéo ou erreur de lecture.")
            break

        # Faire une détection d'objets avec YOLOv5s
        results = model(frame)

        # Récupérer les résultats de détection
        for *xyxy, conf, class_id in results.xyxy[0]:  # Liste des objets détectés
            x1, y1, x2, y2 = map(int, xyxy)  # Coordonnées du rectangle
            class_id = int(class_id)  # ID de la classe détectée
            conf = float(conf)  # Score de confiance

            width, height = x2 - x1, y2 - y1  # Calcul des dimensions
            class_name = model.names[class_id]  # Nom de la classe détectée

            # Vérifier si le colis est valide
            is_valid = (
                class_name in VALID_CLASSES and
                MIN_WIDTH <= width <= MAX_WIDTH and
                MIN_HEIGHT <= height <= MAX_HEIGHT
            )

            # Définir la couleur en fonction de la validation
            color = (0, 255, 0) if is_valid else (0, 0, 255)  # Vert = valide, Rouge = invalide
            label = f"{class_name} ({conf:.2f})"

            # Dessiner le rectangle et le label sur l'image
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Enregistrer la frame annotée si un fichier de sortie est spécifié
        if out:
            out.write(frame)

        # Afficher la vidéo en direct
        cv2.imshow('YOLOv5s Colis Detection', frame)

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
