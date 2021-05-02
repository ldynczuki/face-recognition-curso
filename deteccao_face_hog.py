import cv2
import dlib

imagem = cv2.imread("imagens-e-recursos/fotos/grupo.0.jpg")

# Criando o detector de faces
detector = dlib.get_frontal_face_detector()

# Variável para armazenar os bounding box
# Perceba que serão armazenados valores referentes aos pontos das faces detectadas
# Veja que o formato dos valores no dlib é um pouco diferente do OpenCV, isso ocorre, como vimos anteriormente,
# porque o dlib encontra 2 pontos para detectar face, formato: [(ponto superior esquerdo) e (ponto inferior direito)]
# Utilizamos o parâmetro 1 para aumentar a imagem 1 vez para melhorar a detecção, é muito bom para imagens pequenas
# Por default o valor é 0
facesDetectadas = detector(imagem, 1)
print(facesDetectadas)
print("Faces Detectadas", len(facesDetectadas))

# Criando um for para iterar sobre os valores dos pontos das faces detectadas
for face in facesDetectadas:
    # Exibindo os valores dos pontos das faces, primeiramente os pontos completos e posteriormente separadamente
    # print(face)
    # print(face.left())
    # print(face.top())
    # print(face.right())
    # print(face.bottom())

    # Criando as variáveis l (left), t (top), r (right), b (bottom) para armazenar os respectivos valores das faces
    l, t, r, b = (int(face.left()), int(face.top()), int(face.right()), int(face.bottom()))

    # Agora iremos realizar o desenho do retângulo na imagem utilizando Open CV
    cv2.rectangle(imagem, (l, t), (r, b), (0, 255, 255), 2)

cv2.imshow("Detector HOG", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()
