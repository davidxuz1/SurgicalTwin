import PIL.Image as Image
import cv2
import numpy as np
import os
from stages.dehaze.gf import guided_filter

class DehazeStage:
    def __init__(self, omega=0.95, t0=0.1, radius=7, r=20, eps=0.001):
        self.video_processor = VideoHazeRemoval(omega, t0, radius, r, eps)
        
    def process(self, input_video, output_video):
        """
        Process the video and return the output path
        """
        self.video_processor.process_video(input_video, output_video)
        return output_video

class VideoHazeRemoval(object):
    def __init__(self, omega=0.95, t0=0.1, radius=7, r=20, eps=0.001):
        self.omega = omega
        self.t0 = t0
        self.radius = radius
        self.r = r
        self.eps = eps

    def enhance_visibility(self, alpha=1.5, beta=30):
        self.dst = cv2.convertScaleAbs(self.dst, alpha=alpha, beta=beta)

    def get_dark_channel(self, radius=7):
        tmp = self.src.min(axis=2)
        for i in range(self.rows):
            for j in range(self.cols):
                rmin = max(0,i-radius)
                rmax = min(i+radius,self.rows-1)
                cmin = max(0,j-radius)
                cmax = min(j+radius,self.cols-1)
                self.dark[i,j] = tmp[rmin:rmax+1,cmin:cmax+1].min()

    def get_air_light(self):
        flat = self.dark.flatten()
        flat.sort()
        num = int(self.rows*self.cols*0.001)
        threshold = flat[-num]
        tmp = self.src[self.dark>=threshold]
        tmp.sort(axis=0)
        self.Alight = tmp[-num:,:].mean(axis=0)

    def get_transmission(self, radius=7, omega=0.95):
        for i in range(self.rows):
            for j in range(self.cols):
                rmin = max(0,i-radius)
                rmax = min(i+radius,self.rows-1)
                cmin = max(0,j-radius)
                cmax = min(j+radius,self.cols-1)
                pixel = (self.src[rmin:rmax+1,cmin:cmax+1]/self.Alight).min()
                self.tran[i,j] = 1. - omega * pixel

    def guided_filter(self, r=60, eps=0.001):
        self.gtran = guided_filter(self.src, self.tran, r, eps)

    def recover(self, t0=0.1):
        self.gtran[self.gtran<t0] = t0
        t = self.gtran.reshape(*self.gtran.shape,1).repeat(3,axis=2)
        self.dst = (self.src.astype(np.double) - self.Alight)/t + self.Alight
        self.dst *= 255
        self.dst[self.dst>255] = 255
        self.dst[self.dst<0] = 0
        self.dst = self.dst.astype(np.uint8)

    def process_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print("Error: No se pudo abrir el video")
            return

        # Obtener propiedades del video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Configurar el writer para el video de salida
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Procesar frame
            self.src = frame.astype(np.double)/255.
            self.rows, self.cols, _ = self.src.shape
            self.dark = np.zeros((self.rows, self.cols), dtype=np.double)
            self.Alight = np.zeros(3, dtype=np.double)
            self.tran = np.zeros((self.rows, self.cols), dtype=np.double)
            self.dst = np.zeros_like(self.src, dtype=np.double)

            # Aplicar algoritmo de dehaze
            self.get_dark_channel()
            self.get_air_light()
            self.get_transmission()
            self.guided_filter()
            self.recover()
            self.enhance_visibility()

            # Guardar frame procesado
            out.write(self.dst)
            
            # Mostrar progreso
            frame_count += 1
            print(f"Procesando frame {frame_count}/{total_frames}")

        # Liberar recursos
        cap.release()
        out.release()
        print("Procesamiento de video completado")