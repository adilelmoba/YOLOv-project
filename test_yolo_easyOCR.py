"""
Ce script analyse une vidéo en utilisant le modèle YOLOv5.
Il utilise PyTorch, OpenCV et EasyOCR pour la détection en temps réel.
Les résultats OCR sont mis à jour toutes les n frames et restent affichés jusqu'à la prochaine analyse.
"""

import argparse
import warnings

import cv2
import easyocr
import torch

warnings.filterwarnings("ignore", category=FutureWarning)


def draw_bounding_box_with_text(frame, bbox, text):
    """Dessine un cadre et affiche le texte sur le frame."""
    x_min, y_min, x_max, y_max = bbox
    # Dessiner le rectangle
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    # Afficher le texte au-dessus du rectangle
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

    # Charger le modèle YOLO (une seule fois)
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

    # Déterminer la source vidéo
    cap = cv2.VideoCapture(args.video if args.video else args.camera)

    # Configurer l'enregistrement si spécifié
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # Initialiser EasyOCR avec support GPU si disponible
    reader = easyocr.Reader(["en"], gpu=True)

    frame_count = 0
    # Liste persistante des détections OCR sous forme de tuples (bbox, texte)
    persistent_detections = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Mise à jour des détections OCR toutes les 'freq' frames
        if frame_count % args.freq == 0:
            # Exécuter YOLO sur la frame
            results = model(frame)
            # Utilisation de results.xyxy pour plus de rapidité
            detections = results.xyxy[0].cpu().numpy()
            # Réinitialiser les détections persistantes
            persistent_detections = []

            for detection in detections:
                # Les 6 premières valeurs : xmin, ymin, xmax, ymax, confidence, class_id
                x_min, y_min, x_max, y_max, confidence, class_id = detection[:6]
                # Optionnel : filtrer par confiance minimale (ex : 0.3)
                if confidence < 0.3:
                    continue

                bbox = (int(x_min), int(y_min), int(x_max), int(y_max))
                # Découper la région détectée
                detected_region = frame[bbox[1] : bbox[3], bbox[0] : bbox[2]]
                if detected_region.size == 0:
                    continue

                # Conversion en RGB pour EasyOCR
                detected_region_rgb = cv2.cvtColor(detected_region, cv2.COLOR_BGR2RGB)
                ocr_results = reader.readtext(detected_region_rgb)

                # Combiner tous les textes reconnus dans la région
                texts = [res[1] for res in ocr_results if res[1].strip() != ""]
                if texts:
                    combined_text = " ".join(texts)
                    persistent_detections.append((bbox, combined_text))

        # Afficher les détections persistantes sur la frame
        for bbox, text in persistent_detections:
            draw_bounding_box_with_text(frame, bbox, text)

        # Écrire dans le fichier de sortie si spécifié
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
