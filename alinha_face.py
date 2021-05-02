import cv2
import dlib
import numpy as np


def imprimePontos(imagem, pontosFaciais):
    for p in pontosFaciais.parts():
        cv2.circle(imagem, (p.x, p.y), 2, (0, 255, 0), 2)


detectorFace = dlib.get_frontal_face_detector()
detectorPontos = dlib.shape_predictor("imagens-e-recursos/recursos/shape_predictor_5_face_landmarks.dat")

imagem = cv2.imread("imagens-e-recursos/fotos/treinamento/ronald.0.1.jpg")
# transformando a ordem dos canais de BGR padrão do OpenCV para RGB
imagemRBG = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
facesDetectadas = detectorFace(imagemRBG, 0)
# A variável abaixo irá armazenar os pontos faciais e utilizada também posteriormente para o alinhamento facial
facesPontos = dlib.full_object_detections()

for face in facesDetectadas:
    pontos = detectorPontos(imagemRBG, face)
    print(pontos.parts())
    facesPontos.append(pontos)
    imprimePontos(imagem, pontos)


# Neste momento iremos realizar o alinhamento facial
# Este processo captura a imagem que estamos analisando e irá rotacionar de forma que ela fique "reta", ou seja,
# com os olhos retos. Isto é utilizado muitas vezes ANTES da aplicação do algoritmo de distância (KNN) em
# reconhecimento facial. Inclusive, é indicado utilizar este processo de alinhamento facial antes de realizar
# a classificação de uma imagem, pois irá gerar o descritor facial com os 128 pontos já com o alinhamento facial
# e fará o comparativo com a base já alinhada, o que pode trazer melhores resultados.

# O método get_face_chips irá ter os parâmetros uma imagem e um objeto do tipo full object detections
# e irá retornar as faces em um array do tipo numpy representando a imagem
# a imagem será rotacionada para cima e para a direita e será transformada na escala 150x150 pixels
# basicamente é este método que irá fazer o alinhamento das faces.
imagens = dlib.get_face_chips(imagemRBG, facesPontos)

# Como trabalhamos com a imagem em canais RGB precisamos devolver para BGR para o open cv entender
for img in imagens:
    imagemBGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("Imagem Original", imagem)
    cv2.waitKey(0)
    cv2.imshow("Imagem alinhada", imagemBGR)
    cv2.waitKey(0)


# cv2.imshow("5 pontos", imagem)
# cv2.waitKey(0)

cv2.destroyAllWindows()
