import cv2
import dlib
import numpy as np


# Função para imprimir os pontos nas faces detectadas
def imprimePontos(imagem, pontosFaciais):
    for p in pontosFaciais.parts():
        cv2.circle(imagem, (p.x, p.y), 2, (0, 255, 0), 2)


# Função para imprimir os índices iterados de onde estão posicionados os pontos faciais
def imprimeNumeros(imagem, pontosFaciais):
    for i, p in enumerate(pontosFaciais.parts()):
        cv2.putText(imagem, str(i), (p.x, p.y), fonte, .55, (0, 0, 255), 1)


# Função para imprimir linhas ligadas pelos pontos faciais detectados
def imprimeLinhas(imagem, pontosFaciais):
    # o parâmetro False indica que a linha será ligada apenas uma vez e não será fechada do início ao fim
    p68 = [[0, 16, False],   # linha do queixo
           [17, 21, False],  # sobrancelha direita
           [22, 26, False],  # sobrancelha esquerda
           [27, 30, False],  # ponte nasal
           [30, 35, True],   # nariz inferior
           [36, 41, True],   # olho esquerdo
           [42, 47, True],   # olho direito
           [48, 59, True],   # labio externo
           [60, 67, True]]   # labio interno

    for k in range(0, len(p68)):
        pontos = []

        for i in range(p68[k][0], p68[k][1] + 1):
            ponto = [pontosFaciais.part(i).x, pontosFaciais.part(i).y]
            pontos.append(ponto)

        pontos = np.array(pontos, dtype=np.int32)
        cv2.polylines(imagem, [pontos], p68[k][2], (255, 0, 0), 2)


fonte = cv2.FONT_HERSHEY_COMPLEX_SMALL
# imagem = cv2.imread("imagens-e-recursos/fotos/treinamento/ronald.0.0.jpg")
# imagem = cv2.imread("imagens-e-recursos/fotos/treinamento/ronald.0.1.jpg")
# imagem = cv2.imread("imagens-e-recursos/fotos/treinamento/ronald.0.2.jpg")
imagem = cv2.imread("imagens-e-recursos/fotos/treinamento/ronald.0.3.jpg")
# imagem = cv2.imread("imagens-e-recursos/fotos/treinamento/ronald.0.4.jpg")


# Criação dos classificadores (detector de faces e de pontos faciais)
detectorFace = dlib.get_frontal_face_detector()
detectorPontos = dlib.shape_predictor("imagens-e-recursos/recursos/shape_predictor_68_face_landmarks.dat")

# armazenando os valores do bounding boxes das faces detectadas
facesDetectadas = detectorFace(imagem, 2)

# Percorrer por cada face detectada e armazena os 68 pontos faciais
for face in facesDetectadas:
    # Utilizando o detector de pontos e vamos passar a imagem original e o ROI (bounding boxes - pontos de interesse)
    pontos = detectorPontos(imagem, face)
    # Exibe os valores dos 68 pontos - é retornado uma lista com tuplas de valores nas posições (x, y) cada ponto é x, y
    print(pontos.parts())
    # Sempre deverá retornar o valor de 68 (pontos), pq o dlib tem a premissa de que só é uma face se tiver os 68 pontos
    print(len(pontos.parts()))
    # Chama a função imprimePontos
    # imprimePontos(imagem, pontos)
    # Chama a função imprimeNumeros
    # imprimeNumeros(imagem, pontos)
    # Chama a função imprimeLinhas
    imprimeLinhas(imagem, pontos)

cv2.imshow("Pontos Faciais", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()
