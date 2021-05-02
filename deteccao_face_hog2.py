import cv2
import dlib

# Criação do vetor subdetector para auxiliar a qual subdetector foi utilizado para detectar face.
# Esses são os 5 subdetectores (e nessa ordem) que existem no dlib e que é utilizado internamente.
subdetector = ["Olhar a frente", "Vista a esquerda", "Vista a direita", "A frente girando a esquerda",
               "A frente girando a direita"]

imagem = cv2.imread("imagens-e-recursos/fotos/grupo.0.jpg")

# Criando a variável para receber o método de detectar faces
detector = dlib.get_frontal_face_detector()

# Criamos as variáveis para o armazenamento dos pontos das faces detectadas
# Além disso, criamos as variáveis "pontuacao" que irá armazenar qual é a confiabilidade da classificação e
# na variável idx vai armazenar qual foi o subdetector que foi utilizado (para qual lado a face está "virada")
# Utilizamos o parâmetro 1 para aumentar a imagem 1 vez para melhorar a detecção, é muito bom para imagens pequenas
# Por default o valor é 0
# Outro parâmetro é o limiar de faces detectadas, se o valor for negativo, vai aumentar a quantidade de faces detectada
# Todavia, pode gerar erros nas faces detectadas: detector.run(imagem, 1, -1)
facesDetectadas, pontuacao, idx = detector.run(imagem, 1)

# Exibe os pontos das faces detectadas
# print(facesDetectadas)

# Exibe os valores de confiabilidade das faces detectadas, quanto maior o valor, mais as chances de realmente ser face.
# print(pontuacao)

# Exibe qual foi o subdetector utilizado internamente para detectar as faces - a lista auxilia a leitura do subdetector.
# print(idx)

# Criei um for enumerate para iterar os valores das faces e também os seus índices
for i, d in enumerate(facesDetectadas):
    # print(i)
    # print(d)
    print("Detecção: {}, pontuação: {}, Sub-detector: {}".format(d, pontuacao[i], subdetector[i]))

    # Armazenando os valores dos pontos das faces detectadas
    l, t, r, b = (int(d.left()), int(d.top()), int(d.right()), int(d.bottom()))

    # Criando o desenho do retângulo nas faces detectadas
    cv2.rectangle(imagem, (l, t), (r, b), (0, 255, 0), 2)

cv2.imshow("Detector HOG2", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()