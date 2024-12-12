import cv2
import os
import sys
import argparse

# Obtener la ruta del proyecto
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def parse_arguments():
    parser = argparse.ArgumentParser(description='Seleccionar bounding box de un video o imagen')
    parser.add_argument('--input', type=str, 
                        default=os.path.join(PROJECT_ROOT, "data/input/video.mp4"),
                        help='Ruta al archivo de video o imagen')
    return parser.parse_args()

def main():
    args = parse_arguments()
    file_path = args.input

    if not os.path.exists(file_path):
        print(f"Error: El archivo {file_path} no existe.")
        sys.exit(1)

    # Determinar si el archivo es un video o una imagen
    file_extension = os.path.splitext(file_path)[1].lower()

    # Leer el primer cuadro (imagen o primer frame del video)
    frame = None
    if file_extension == ".mp4":
        cap = cv2.VideoCapture(file_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("Error al cargar el video")
            sys.exit(1)
    elif file_extension in [".jpg", ".jpeg", ".png"]:
        frame = cv2.imread(file_path)
        if frame is None:
            print("Error al cargar la imagen")
            sys.exit(1)
    else:
        print("Error: Formato no soportado. Usa un archivo .mp4, .jpg, .jpeg o .png.")
        sys.exit(1)

    # Seleccionar bounding box manualmente
    bbox = cv2.selectROI("Selecciona el bounding box", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    x, y, w, h = bbox
    xmin, ymin, xmax, ymax = int(x), int(y), int(x+w), int(y+h)

    # Crear directorio de salida si no existe
    output_dir = os.path.join(PROJECT_ROOT, "data/tools_output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar las coordenadas en formato xmin,ymin,xmax,ymax en un archivo .txt
    output_path = os.path.join(output_dir, "fixed_bbox_watermark.txt")
    with open(output_path, "w") as f:
        f.write(f"{xmin} {ymin} {xmax} {ymax}\n")

    # Imprimir el resultado
    print(f"Bounding box guardado en directorio: {output_dir}")
    print(f"Coordenadas: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")

if __name__ == "__main__":
    main()
