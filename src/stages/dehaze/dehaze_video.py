import cv2
import numpy as np
from numba import jit

class DehazeParams:
    def __init__(self):
        self.omega = 100    # Intensidad dehaze (50-95 recomendado) - Mayor valor = más eliminación de neblina
        self.t0 = 58       # Preservación detalles - Mayor valor = más detalles preservados
        self.radius = 2    # Radio dark channel (3-15) - Mayor valor = más precisión pero más lento
        self.r = 21        # Kernel gaussiano (3-31, impar) - Mayor valor = más suavizado
        self.alpha = 181   # Contraste (100-200) - Mayor valor = más contraste
        self.beta = 10      # Brillo (0-50) - Mayor valor = más brillo

        
    def get_omega(self):
        return self.omega / 100.0
        
    def get_t0(self):
        return self.t0 / 100.0
        
    def get_alpha(self):
        return self.alpha / 100.0

class DehazeStage:
    def __init__(self):
        self.params = DehazeParams()
        self.video_processor = VideoHazeRemoval(self.params)
        
    def process(self, input_video, output_video):
        self.video_processor.process_video(input_video, output_video)
        return output_video

@jit(nopython=True)
def min_channel(src):
    rows, cols, channels = src.shape
    result = np.zeros((rows, cols), dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            result[i, j] = min(src[i, j, 0], src[i, j, 1], src[i, j, 2])
    return result

@jit(nopython=True)
def get_min_local_patch(arr, radius):
    rows, cols = arr.shape
    result = np.zeros((rows, cols), dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            min_val = arr[i, j]
            for di in range(max(0, i-radius), min(rows, i+radius+1)):
                for dj in range(max(0, j-radius), min(cols, j+radius+1)):
                    if arr[di, dj] < min_val:
                        min_val = arr[di, dj]
            result[i, j] = min_val
    return result

class VideoHazeRemoval:
    def __init__(self, params):
        self.params = params
        self.paused = False

    @staticmethod
    @jit(nopython=True)
    def _get_dark_channel(src, radius):
        min_rgb = min_channel(src)
        return get_min_local_patch(min_rgb, radius)

    def process_frame(self, frame):
        src = frame.astype(np.float64)/255.
        rows, cols = src.shape[:2]
        
        dark = self._get_dark_channel(src, self.params.radius)
        
        flat_idx = np.argsort(dark.ravel())[-int(dark.size * 0.001):]
        flat_coords = np.unravel_index(flat_idx, dark.shape)
        Alight = src[flat_coords].mean(axis=0)
        Alight = Alight.reshape(1, 1, 3)
        
        tran = np.ones((rows, cols), dtype=np.float64)
        for i in range(rows):
            for j in range(cols):
                tran[i, j] = 1 - self.params.get_omega() * dark[i, j]
        
        tran = cv2.GaussianBlur(tran.astype(np.float32), (self.params.r, self.params.r), 0)
        tran = np.clip(tran, self.params.get_t0(), 1)
        tran = np.expand_dims(tran, axis=2)
        
        dst = ((src - Alight)/tran + Alight) * 255
        dst = np.clip(dst, 0, 255).astype(np.uint8)
        
        dst = cv2.convertScaleAbs(dst, alpha=self.params.get_alpha(), beta=self.params.beta)
        
        return dst

    def process_video(self, input_path, output_path):
        def on_trackbar(x):
            pass

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print("Error: No se pudo abrir el video")
            return

        # Crear ventanas de control
        cv2.namedWindow('Dehaze Controls')
        cv2.namedWindow('Original')
        cv2.namedWindow('Processed')
        
        # Crear trackbars
        cv2.createTrackbar('Omega x100', 'Dehaze Controls', self.params.omega, 100, on_trackbar)
        cv2.createTrackbar('T0 x100', 'Dehaze Controls', self.params.t0, 100, on_trackbar)
        cv2.createTrackbar('Radius', 'Dehaze Controls', self.params.radius, 15, on_trackbar)
        cv2.createTrackbar('Kernel Size', 'Dehaze Controls', self.params.r, 31, on_trackbar)
        cv2.createTrackbar('Alpha x100', 'Dehaze Controls', self.params.alpha, 200, on_trackbar)
        cv2.createTrackbar('Beta', 'Dehaze Controls', self.params.beta, 100, on_trackbar)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        ret, frame = cap.read()
        
        print("Controles:")
        print("'p' - Pausar/Reanudar")
        print("'q' - Salir")
        print("Mientras está pausado, ajusta los parámetros y presiona 'p' para continuar")

        while ret:
            if not self.paused:
                ret, frame = cap.read()
                if not ret:
                    break
            
            # Actualizar parámetros desde trackbars
            self.params.omega = cv2.getTrackbarPos('Omega x100', 'Dehaze Controls')
            self.params.t0 = cv2.getTrackbarPos('T0 x100', 'Dehaze Controls')
            self.params.radius = max(1, cv2.getTrackbarPos('Radius', 'Dehaze Controls'))
            self.params.r = max(1, cv2.getTrackbarPos('Kernel Size', 'Dehaze Controls'))
            if self.params.r % 2 == 0:
                self.params.r += 1
            self.params.alpha = cv2.getTrackbarPos('Alpha x100', 'Dehaze Controls')
            self.params.beta = cv2.getTrackbarPos('Beta', 'Dehaze Controls')
            
            processed_frame = self.process_frame(frame)
            
            # Mostrar frames
            cv2.imshow('Original', frame)
            cv2.imshow('Processed', processed_frame)
            
            if not self.paused:
                out.write(processed_frame)
                frame_count += 1
                print(f"Procesando frame {frame_count}/{total_frames}")

            # Control de teclas
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                self.paused = not self.paused
                status = "PAUSADO" if self.paused else "REPRODUCIENDO"
                print(f"\nEstado: {status}")
                if self.paused:
                    print("Ajusta los parámetros y presiona 'p' para continuar")

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print("\nParámetros finales utilizados:")
        print(f"Omega: {self.params.get_omega():.2f}")
        print(f"T0: {self.params.get_t0():.2f}")
        print(f"Radius: {self.params.radius}")
        print(f"Kernel Size: {self.params.r}")
        print(f"Alpha: {self.params.get_alpha():.2f}")
        print(f"Beta: {self.params.beta}")


# import cv2
# import numpy as np
# from numba import jit
# import concurrent.futures

# class DehazeStage:
#     def __init__(self, omega=0.95, t0=0.1, radius=7, r=21, eps=0.001):  # r cambiado a 21
#         self.omega = omega
#         self.t0 = t0
#         self.radius = radius
#         self.r = r
#         self.eps = eps

#         self.video_processor = VideoHazeRemoval(omega, t0, radius, r, eps)
        
#     def process(self, input_video, output_video):
#         self.video_processor.process_video(input_video, output_video)
#         return output_video

# @jit(nopython=True)
# def min_channel(src):
#     rows, cols, channels = src.shape
#     result = np.zeros((rows, cols), dtype=np.float64)
#     for i in range(rows):
#         for j in range(cols):
#             result[i, j] = min(src[i, j, 0], src[i, j, 1], src[i, j, 2])
#     return result

# @jit(nopython=True)
# def get_min_local_patch(arr, radius):
#     rows, cols = arr.shape
#     result = np.zeros((rows, cols), dtype=np.float64)
#     for i in range(rows):
#         for j in range(cols):
#             min_val = arr[i, j]
#             for di in range(max(0, i-radius), min(rows, i+radius+1)):
#                 for dj in range(max(0, j-radius), min(cols, j+radius+1)):
#                     if arr[di, dj] < min_val:
#                         min_val = arr[di, dj]
#             result[i, j] = min_val
#     return result

# class VideoHazeRemoval:
#     def __init__(self, omega=0.95, t0=0.1, radius=7, r=20, eps=0.001):
#         self.omega = omega
#         self.t0 = t0
#         self.radius = radius
#         self.r = r
#         self.eps = eps

#     @staticmethod
#     @jit(nopython=True)
#     def _get_dark_channel(src, radius):
#         min_rgb = min_channel(src)
#         return get_min_local_patch(min_rgb, radius)

#     def process_frame(self, frame):
#         src = frame.astype(np.float64)/255.
#         rows, cols = src.shape[:2]
        
#         # Obtener dark channel
#         dark = self._get_dark_channel(src, self.radius)
        
#         # Obtener luz atmosférica
#         flat_idx = np.argsort(dark.ravel())[-int(dark.size * 0.001):]
#         flat_coords = np.unravel_index(flat_idx, dark.shape)
#         Alight = src[flat_coords].mean(axis=0)
        
#         # Obtener mapa de transmisión y aplicar filtro gaussiano
#         tran = np.ones((rows, cols), dtype=np.float64)
#         for i in range(rows):
#             for j in range(cols):
#                 tran[i, j] = 1 - self.omega * dark[i, j]
        
#         # Usar filtro gaussiano en lugar de guided filter
#         tran = cv2.GaussianBlur(tran.astype(np.float32), (self.r, self.r), 0)
        
#         # Recuperar imagen
#         tran = np.clip(tran, self.t0, 1)
#         tran = np.expand_dims(tran, axis=2).repeat(3, axis=2)
#         dst = ((src - Alight)/tran + Alight) * 255
#         dst = np.clip(dst, 0, 255).astype(np.uint8)
        
#         # Mejorar visibilidad
#         dst = cv2.convertScaleAbs(dst, alpha=1.5, beta=30)
#         return dst


#     def process_video(self, input_path, output_path):
#         cap = cv2.VideoCapture(input_path)
#         if not cap.isOpened():
#             print("Error: No se pudo abrir el video")
#             return

#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         fps = int(cap.get(cv2.CAP_PROP_FPS))
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#         frame_count = 0
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
                
#             processed_frame = self.process_frame(frame)
#             out.write(processed_frame)
            
#             frame_count += 1
#             print(f"Procesando frame {frame_count}/{total_frames}")

#         cap.release()
#         out.release()
#         print("Procesamiento de video completado")



# import PIL.Image as Image
# import cv2
# import numpy as np
# import os
# from stages.dehaze.gf import guided_filter

# class DehazeStage:
#     def __init__(self, omega=0.95, t0=0.1, radius=7, r=20, eps=0.001):
#         self.video_processor = VideoHazeRemoval(omega, t0, radius, r, eps)
        
#     def process(self, input_video, output_video):
#         """
#         Process the video and return the output path
#         """
#         self.video_processor.process_video(input_video, output_video)
#         return output_video

# class VideoHazeRemoval(object):
#     def __init__(self, omega=0.95, t0=0.1, radius=7, r=20, eps=0.001):
#         self.omega = omega
#         self.t0 = t0
#         self.radius = radius
#         self.r = r
#         self.eps = eps

#     def enhance_visibility(self, alpha=1.5, beta=30):
#         self.dst = cv2.convertScaleAbs(self.dst, alpha=alpha, beta=beta)

#     def get_dark_channel(self, radius=7):
#         tmp = self.src.min(axis=2)
#         for i in range(self.rows):
#             for j in range(self.cols):
#                 rmin = max(0,i-radius)
#                 rmax = min(i+radius,self.rows-1)
#                 cmin = max(0,j-radius)
#                 cmax = min(j+radius,self.cols-1)
#                 self.dark[i,j] = tmp[rmin:rmax+1,cmin:cmax+1].min()

#     def get_air_light(self):
#         flat = self.dark.flatten()
#         flat.sort()
#         num = int(self.rows*self.cols*0.001)
#         threshold = flat[-num]
#         tmp = self.src[self.dark>=threshold]
#         tmp.sort(axis=0)
#         self.Alight = tmp[-num:,:].mean(axis=0)

#     def get_transmission(self, radius=7, omega=0.95):
#         for i in range(self.rows):
#             for j in range(self.cols):
#                 rmin = max(0,i-radius)
#                 rmax = min(i+radius,self.rows-1)
#                 cmin = max(0,j-radius)
#                 cmax = min(j+radius,self.cols-1)
#                 pixel = (self.src[rmin:rmax+1,cmin:cmax+1]/self.Alight).min()
#                 self.tran[i,j] = 1. - omega * pixel

#     def guided_filter(self, r=60, eps=0.001):
#         self.gtran = guided_filter(self.src, self.tran, r, eps)

#     def recover(self, t0=0.1):
#         self.gtran[self.gtran<t0] = t0
#         t = self.gtran.reshape(*self.gtran.shape,1).repeat(3,axis=2)
#         self.dst = (self.src.astype(np.double) - self.Alight)/t + self.Alight
#         self.dst *= 255
#         self.dst[self.dst>255] = 255
#         self.dst[self.dst<0] = 0
#         self.dst = self.dst.astype(np.uint8)

#     def process_video(self, input_path, output_path):
#         cap = cv2.VideoCapture(input_path)
#         if not cap.isOpened():
#             print("Error: No se pudo abrir el video")
#             return

#         # Obtener propiedades del video
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         fps = int(cap.get(cv2.CAP_PROP_FPS))
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#         # Configurar el writer para el video de salida
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#         frame_count = 0
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # Procesar frame
#             self.src = frame.astype(np.double)/255.
#             self.rows, self.cols, _ = self.src.shape
#             self.dark = np.zeros((self.rows, self.cols), dtype=np.double)
#             self.Alight = np.zeros(3, dtype=np.double)
#             self.tran = np.zeros((self.rows, self.cols), dtype=np.double)
#             self.dst = np.zeros_like(self.src, dtype=np.double)

#             # Aplicar algoritmo de dehaze
#             self.get_dark_channel()
#             self.get_air_light()
#             self.get_transmission()
#             self.guided_filter()
#             self.recover()
#             self.enhance_visibility()

#             # Guardar frame procesado
#             out.write(self.dst)
            
#             # Mostrar progreso
#             frame_count += 1
#             print(f"Procesando frame {frame_count}/{total_frames}")

#         # Liberar recursos
#         cap.release()
#         out.release()
#         print("Procesamiento de video completado")