import cv2
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.HandTrackingModule import HandDetector

# Inicializar captura de vídeo
video = cv2.VideoCapture(0)
video.set(3, 640)  # Largura da tela
video.set(4, 480)  # Altura da tela

# Inicializar detectores
face_detector = FaceDetector()
hand_detector = HandDetector(detectionCon=0.75, maxHands=2)  # maximo de 2 mãos

# Variável para armazenar o somatório total de dedos levantados
total_fingers = 0

while True:
    # Ler frame da câmera
    _, img = video.read()

    # Detectar rostos
    img, bboxes = face_detector.findFaces(img, draw=True)

    # Detectar mãos e contar dedos
    hands, img = hand_detector.findHands(img)

    if hands:
        total_fingers = 0  # Reiniciar o total de dedos levantados
        for hand in hands:
            fingers = hand_detector.fingersUp(hand)
            no_of_fingers = fingers.count(1)
            total_fingers += no_of_fingers  # Adicionar ao total de dedos levantados
            cv2.putText(img, f'Dedos: {no_of_fingers}', (hand['lmList'][0][0], hand['lmList'][0][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Desenhar o total de dedos levantados no topo da imagem
    cv2.putText(img, f'Total de dedos: {total_fingers}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar imagem resultante
    cv2.imshow('Resultado', img)

    # Sair do loop se a tecla 'Esc' for pressionada
    if cv2.waitKey(1) == 27:
        break

# Liberar recursos
video.release()
cv2.destroyAllWindows()