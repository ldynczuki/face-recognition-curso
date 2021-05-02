import cv2
import dlib

imagem = cv2.imread("imagens-e-recursos/fotos/grupo.0.jpg")

# Criação do detector de faces do CNN
# É necessário passar como parâmetro um arquivo já treinado
detector = dlib.cnn_face_detection_model_v1("imagens-e-recursos/recursos/mmod_human_face_detector.dat")

# Inserimos o parâmetro 1 para aumentar em 1x o tamanho da imagem. Se não tiver nenhum valor o padrão é 0
# Sem o valor 1 o resultado pode ser degradado.
facesDetectadas = detector(imagem, 1)
print(facesDetectadas)
print("Faces Detectadas", len(facesDetectadas))

# Criando for para iterar sobre as faces detectadas e armazenando para criar os bounding box
for idx, face in enumerate(facesDetectadas):
    # criando as variáveis l (left), t (top), r (right) e b (bottom) para armazenar os valores e c (confiança)
    l, t, r, b, c = (int(face.rect.left()), int(face.rect.top()), int(face.rect.right()),
                     int(face.rect.bottom()), face.confidence)
    print("Face Detectada {} - Confiança: {}".format(idx, c))
    # Inserindo retângulo nas faces detectadas
    cv2.rectangle(imagem, (l, t), (r, b), (255, 0, 0), 2)

cv2.imshow("Detector CNN", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()
