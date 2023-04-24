#Imports
#CV2 es para los graficos (imagenes, video, etc)
import cv2
#mediapipe es para el reconocimiento facial
import mediapipe as mp

#Es el modelo que se usa para reconocer la cara que esta en mediapipe y se llama FaceDetectionModel
mp_face_detection = mp.solutions.face_detection
#Este es para dibujar en las imagenes
mp_drawing = mp.solutions.drawing_utils

#Dar a elegir entre modelo 1 o 2
print("Elige el método de detección de rostro:")
print("1. Imagen")
print("2. Video")
#Aqui se selecciona 1 para imagen y 2 para video
model_selection = int(input())

#Imagen
if model_selection == 1:

    #Variable para contar las caras
    contador = 0

    #Se selecciona el FaceDetection con model 1 para imagenes y confidence en 0.25 (fue el que mejor funcionó)
    with mp_face_detection.FaceDetection(model_selection=1,
        min_detection_confidence=0.25) as face_detection:

        #Se pide la ruta de la imagen (Si esta en la misma carpeta solo poner el nombre con extensión)
        print("Seleccina una imagen para detectar rostros")
        image_path = input()

        #Se lee la imagen
        image = cv2.imread(image_path)
        #Se obtienen sus medidas
        height, width, _ = image.shape
        #La imagen se convierte a RGB porque viene en BGR
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #Se procesa y se guarda en la variable results
        results = face_detection.process(image_rgb)
        
        #Si el resultado no esta vacio se entra en el if
        if results.detections is not None:
            #Se recorren todas las caras detectadas
            for detection in results.detections:
                #Por cada cara se aumenta en 1 el contador
                contador += 1
                #Se pinta un rectangulo alrededor de la cara
                xmin = int(detection.location_data.relative_bounding_box.xmin * width)
                ymin = int(detection.location_data.relative_bounding_box.ymin * height)
                w = int(detection.location_data.relative_bounding_box.width * width)
                h = int(detection.location_data.relative_bounding_box.height * height)
                cv2.rectangle(image, (xmin, ymin), (xmin + w, ymin + h), (0, 255, 0), 3)

        #Se muestra la imagen con los rectangulos y el contador de caras
        cv2.putText(image, "Total de rostros detectados: " + str(contador), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Image", image)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

#Video
else:
    #Variable para contar las caras
    num_faces = 0
    #Variable para la camara
    cap = cv2.VideoCapture(0)

    #Se selecciona el FaceDetection con model 0 para videos y confidence en 0.5 (fue el que mejor funcionó)
    with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5) as face_detection:

        #Mientras la camara este encendida
        while cap.isOpened():
            #Se captura un frame
            success, image = cap.read()
            #Manda error
            if not success:
                print("Sin camara.")
                continue

            #Se obtienen sus medidas
            height, width, _ = image.shape

            #La imagen se convierte a RGB porque viene en BGR
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #Se guarda en la variable results
            results = face_detection.process(image)

            #Se resetea el contador de caras para cada frame
            num_faces = 0

            #No se porque pero no lo muevas, es igual que arriba xD
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            #Si el resultado no esta vacio se entra en el if
            if results.detections:
                #Se recorren todas las caras detectadas
                for detection in results.detections:
                    #Por cada cara se aumenta en 1 el contador
                    num_faces += 1
                    #Se pinta un rectangulo alrededor de la cara
                    xmin = int(detection.location_data.relative_bounding_box.xmin * width)
                    ymin = int(detection.location_data.relative_bounding_box.ymin * height)
                    w = int(detection.location_data.relative_bounding_box.width * width)
                    h = int(detection.location_data.relative_bounding_box.height * height)
                    cv2.rectangle(image, (xmin, ymin), (xmin + w, ymin + h), (0, 255, 0), 1)
            #Se muestra la imagen con los rectangulos y el contador de caras
            cv2.putText(image, "Total de rostros detectados: " + str(num_faces), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Frame", image)
            #Este se cierra apretando control + c en la consola
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()

    