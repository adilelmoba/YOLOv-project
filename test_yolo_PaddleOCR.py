import argparse

import cv2
import torch
from paddleocr import PaddleOCR


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

    # Initialiser PaddleOCR
    ocr = PaddleOCR(lang="en")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Détection avec YOLO
        results = model(frame)
        detections = results.pandas().xyxy[0]
        plate_detections = detections[detections["name"] == "plate"]

        for _, row in plate_detections.iterrows():
            x_min, y_min, x_max, y_max = (
                int(row["xmin"]),
                int(row["ymin"]),
                int(row["xmax"]),
                int(row["ymax"]),
            )
            plate_image = frame[y_min:y_max, x_min:x_max]
            plate_rgb = cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB)
            result = ocr.ocr(plate_rgb, cls=True)

            for line in result[0]:
                print(f"Texte détecté : {line[1][0]}")

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
