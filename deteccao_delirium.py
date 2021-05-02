import dlib
import glob
import cv2
import os

# Variável abaixo está armazenando o objeto da biblioteca dlib
opcoes = dlib.simple_object_detector_training_options()
# O parâmetro TRUE abaixo é muito utilizada caso você esteja treinando o seu próprio detector facial
# No nosso caso atual, é como se "inclinássemos" a imagem do relógio um pouco para o lado para melhorar a detecção.
opcoes.add_left_right_image_flips = True
# O parâmetro C abaixo é o custo do algoritmo SVM
opcoes.C = 5

# Estamos realizando o treinamento do nosso algoritmo, onde passamos os parâmetros de:
# primeiramente as imagens com os bounding boxes que fizemos no imglab e depois o nome do nosso algoritmo com extensão.
# Depois da primeira execução, onde temos nosso arquivo de treinamento criado, podemos comentar a linha abaixo.
# dlib.train_simple_object_detector("imagens-e-recursos/recursos/treinamento_delirium.xml",
#                                  "imagens-e-recursos/recursos/detector_delirium.svm", opcoes)


# A partir deste momento, iremos iniciar as etapas de testes de detecção, onde acima já foi treinado o svm.
detector = dlib.simple_object_detector("imagens-e-recursos/recursos/detector_delirium.svm")

for imagem in glob.glob(os.path.join("imagens-e-recursos/delirium", "*.jpg")):
    img = cv2.imread(imagem)
    objetosDetectados = detector(img, 2)

    for d in objetosDetectados:
        l, t, r, b = (int(d.left()), int(d.top()), int(d.right()), int(d.bottom()))
        cv2.rectangle(img, (l, t), (r, b), (0, 0, 255), 2)

    cv2.imshow("Detector Delirium", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
