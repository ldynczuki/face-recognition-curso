import cv2
import dlib

fonte = cv2.FONT_HERSHEY_COMPLEX_SMALL

# Iremos utilizar as imagens abaixo (uma de cada vez) para testar a confiabilidade dos detectores HOG e CNN
imagem = cv2.imread("imagens-e-recursos/fotos/grupo.0.jpg")
# imagem = cv2.imread("imagens-e-recursos/fotos/grupo.1.jpg")
# imagem = cv2.imread("imagens-e-recursos/fotos/grupo.2.jpg")
# imagem = cv2.imread("imagens-e-recursos/fotos/grupo.3.jpg")
# imagem = cv2.imread("imagens-e-recursos/fotos/grupo.4.jpg")
# imagem = cv2.imread("imagens-e-recursos/fotos/grupo.5.jpg")
# imagem = cv2.imread("imagens-e-recursos/fotos/grupo.6.jpg")
# imagem = cv2.imread("imagens-e-recursos/fotos/grupo.7.jpg")
# imagem = cv2.imread("/home/lucas/Downloads/hugging.jpg")


# Criação do classificador Haar
detectorHaar = cv2.CascadeClassifier("imagens-e-recursos/recursos/haarcascade_frontalface_default.xml")
imgemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
facesDetectadasHaar = detectorHaar.detectMultiScale(imgemCinza, scaleFactor=1.3, minSize=(10, 10))

# Criação do classificador HOG
detectorHOG = dlib.get_frontal_face_detector()
facesDetectadasHOG = detectorHOG(imagem, 2)

# Criação do classificador CNN
detectorCNN = dlib.cnn_face_detection_model_v1("imagens-e-recursos/recursos/mmod_human_face_detector.dat")
facesDetectadasCNN = detectorCNN(imagem, 2)

# Neste momento iremos armazenar os valores dos bounding boxes de cada classificador

# Haar
for (x, y, w, h) in facesDetectadasHaar:
    cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Inserindo texto na imagem - o parâmetro após o texto é o posicionamento do texto
    cv2.putText(imagem, "Haar", (x, y - 5), fonte, 0.5, (0, 255, 0))

# HOG
for face in facesDetectadasHOG:
    l, t, r, b = (int(face.left()), int(face.top()), int(face.right()), int(face.bottom()))
    cv2.rectangle(imagem, (l, t), (r, b), (255, 0, 0), 2)
    cv2.putText(imagem, "HOG", (r, t), fonte, 0.5, (255, 0, 0))

# CNN
for face in facesDetectadasCNN:
    l, t, r, b = (int(face.rect.left()), int(face.rect.top()), int(face.rect.right()), int(face.rect.bottom()))
    cv2.rectangle(imagem, (l, t), (r, b), (0, 0, 255), 2)
    cv2.putText(imagem, "CNN", (r, t), fonte, 0.5, (0, 0, 255))

# Exibindo o resultado após a inserção dos retângulos e dos textos
cv2.imshow("Comparativo Detectores", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()
