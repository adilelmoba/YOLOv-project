"""
Ce script analyse une vidéo en utilisant le modèle YOLOv5.
Il utilise PyTorch et OpenCV pour la détection en temps réel.
"""

import argparse
import warnings

import cv2
import easyocr
import torch

warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    # Configuration des arguments
    parser = argparse.ArgumentParser(description="Analyse vidéo avec YOLO.")
    parser.add_argument(
        "-v",
        "--video",
        type=str,
        help="Chemin de la vidéo source. Utilise la caméra si non spécifié.",
    )
    parser.add_argument("-o", "--output", type=str, help="Chemin du fichier de sortie.")
    parser.add_argument(
        "-c",
        "--camera",
        type=int,
        default=0,
        help="Index de la caméra à utiliser si aucune vidéo n'est fournie.",
    )

    args = parser.parse_args()

    # Charger le modèle YOLO
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

    # Déterminer la source de la vidéo
    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(args.camera)

    # Configurer la sortie vidéo si nécessaire
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # Initialiser EasyOCR
    reader = easyocr.Reader(["en"])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Détection avec YOLO
        results = model(frame)
        detections = results.pandas().xyxy[0]
        # plate_detections = detections[detections["name"] == "plate"]
        for _, row in detections.iterrows():
            x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            detected_region = frame[y_min:y_max, x_min:x_max]

            # Appliquer EasyOCR
            detected_region_rgb = cv2.cvtColor(detected_region, cv2.COLOR_BGR2RGB)
            result = reader.readtext(detected_region_rgb)

            for detection in result:
                print(f"Texte détecté : {detection[1]}")

        if out:
            out.write(frame)
        cv2.imshow("YOLO Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
