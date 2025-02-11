"""
Ce script analyse une vidéo en utilisant le modèle YOLOv5.
Il utilise PyTorch, OpenCV et EasyOCR pour la détection en temps réel.
"""

import argparse
import time
import warnings

import cv2
import easyocr
import torch

warnings.filterwarnings("ignore", category=FutureWarning)


def draw_bounding_box_with_text(frame, bbox, text):
    """Dessine un cadre autour de l'objet détecté et affiche le texte."""
    x_min, y_min, x_max, y_max = bbox

    # Dessiner le cadre de l'objet
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Afficher le texte détecté
    cv2.putText(
        frame,
        text,
        (x_min, y_min - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


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
    parser.add_argument(
        "-f",
        "--freq",
        type=int,
        default=5,
        help="Analyser l'image avec OCR tous les 'n' frames.",
    )

    args = parser.parse_args()

    # Charger le modèle YOLO
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

    # Déterminer la source de la vidéo
    cap = cv2.VideoCapture(args.video if args.video else args.camera)

    # Configurer la sortie vidéo si nécessaire
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # Initialiser EasyOCR
    reader = easyocr.Reader(["en"], gpu=True)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Détection avec YOLO
        results = model(frame)

        # Analyser l'OCR tous les 'n' frames
        if frame_count % args.freq == 0:
            detections = results.xyxy[0].cpu().numpy()

            for detection in detections:
                x_min, y_min, x_max, y_max, confidence, class_id = map(
                    int, detection[:6]
                )
                detected_region = frame[y_min:y_max, x_min:x_max]

                # Appliquer EasyOCR sur la région détectée
                detected_region_rgb = cv2.cvtColor(detected_region, cv2.COLOR_BGR2RGB)
                ocr_results = reader.readtext(detected_region_rgb)



                # Afficher les résultats OCR
                for ocr_result in ocr_results:
                    text_detected = ocr_result[1]
                    print(f"Texte détecté : {text_detected}")
                    draw_bounding_box_with_text(
                        frame, (x_min, y_min, x_max, y_max), text_detected
                    )

        # Écrire la vidéo de sortie si spécifiée
        if out:
            out.write(frame)

        # Afficher la vidéo en direct
        cv2.imshow("YOLO Detection", frame)

        # Quitter avec la touche 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Libérer les ressources
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
