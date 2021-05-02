import cv2

imagem = cv2.imread("imagens-e-recursos/fotos/grupo.0.jpg")

# Cria instância do classificador Haar Cascade
classificador = cv2.CascadeClassifier("imagens-e-recursos/recursos/haarcascade_frontalface_default.xml")

# Transformando a imagem original em escala de cinza
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Variável irá receber o método que irá detectar as faces
# Inserindo o parâmetro "scaleFactor" com maiores valores a tendência é melhorar na detecção de faces e vice versa
facesDetectadas = classificador.detectMultiScale(imagemCinza, scaleFactor=1.2, minSize=(50, 50))

# Exibe uma matriz com os valores referente às faces detectadas (onde cada vetor é referente aos valores de uma face)
# Exemplo: [503, 259, 66, 66] => Haarcascade: left = 503, top = 259, width = 66, height = 66
# Obs: pode acontecer de o classificador detectar mais faces na imagem do que realmente existe
print(facesDetectadas)

# Exibe a quantidade de faces detectadas
print("Faces Detectadas", len(facesDetectadas))

# Inserindo o bounding box na imagem
for (x, y, l, a) in facesDetectadas:
    cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 255, 0), 2)

# Nota: os parâmetros do método rectangle são: a imagem a ser desenhada, ponto 1, ponto 2, cores BGR e tamanho da linha

# Exibe a imagem lida anteriormente, aguarda até apertar o 0 e depois destroi as janelas abertas
cv2.imshow("Detector Haar", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("Imagem Cinza", imagemCinza)
cv2.waitKey(0)
cv2.destroyAllWindows()
