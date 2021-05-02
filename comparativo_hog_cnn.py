import cv2
import dlib

# Iremos utilizar as imagens abaixo (uma de cada vez) para testar a confiabilidade dos detectores HOG e CNN
# imagem = cv2.imread("imagens-e-recursos/fotos/grupo.0.jpg")
# imagem = cv2.imread("imagens-e-recursos/fotos/grupo.1.jpg")
imagem = cv2.imread("imagens-e-recursos/fotos/grupo.2.jpg")
# imagem = cv2.imread("imagens-e-recursos/fotos/grupo.3.jpg")
# imagem = cv2.imread("imagens-e-recursos/fotos/grupo.4.jpg")
# imagem = cv2.imread("imagens-e-recursos/fotos/grupo.5.jpg")
# imagem = cv2.imread("imagens-e-recursos/fotos/grupo.6.jpg")
# imagem = cv2.imread("imagens-e-recursos/fotos/grupo.7.jpg")

# Criando o detector de faces HOG
detectorHOG = dlib.get_frontal_face_detector()

# Criando variáveis para armazenar os pontos das faces, a pontuação (confiabilidade) e o subdetector utilizado (idx)
# Utilizando parâmetro 2 para aumentar em 2x a escala da imagem
facesDetectadasHOG, pontuacao, idx = detectorHOG.run(imagem, 2)

# Criando o detector de faces CNN
detectorCNN = dlib.cnn_face_detection_model_v1("imagens-e-recursos/recursos/mmod_human_face_detector.dat")

# Criando variáveis para armazenar os pontos das faces
# Utilizando parâmetro 2 para aumentar em 2x a escala da imagem (como estamos comparando, devemos utilizar igual)
facesDetectadasCNN = detectorCNN(imagem, 2)

# Percorrendo os bounding boxes das faces detectadas do HOG e exibindo a pontuação de confiabilidade
for i, d in enumerate(facesDetectadasHOG):
    print("Pontuação de Confiabilidade HOG {}".format(pontuacao[i]))

print("")

# Percorrendo os bounding boxes das faces detectadas do HOG e exibindo a pontuação de confiabilidade
for face in facesDetectadasCNN:
    print("Pontuação de Confiabilidade CNN {}".format(face.confidence))

cv2.imshow("Comparativo HOG x CNN", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()
