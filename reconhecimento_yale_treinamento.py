import os
import glob
import _pickle as cPickle
import cv2
import dlib
import numpy as np
from PIL import Image

detectorFace = dlib.get_frontal_face_detector()
detectorPontos = dlib.shape_predictor("imagens-e-recursos/recursos/shape_predictor_68_face_landmarks.dat")
reconhecimentoFacial = dlib.face_recognition_model_v1("imagens-e-recursos/recursos/dlib_face_recognition_resnet_model_v1.dat")

indice = {}
idx = 0
descritoresFaciais = None

for arquivo in glob.glob(os.path.join("imagens-e-recursos/yalefaces/treinamento", "*.gif")):
    # Como as imagens do yalefaces são do tipo gif, utilizaremos o pacote Image ao invés do cv2 para carregar
    imagemFace = Image.open(arquivo).convert('RGB')
    imagem = np.array(imagemFace, 'uint8')
    facesDetectadas = detectorFace(imagem, 1)
    numeroFacesDetectadas = len(facesDetectadas)
    # print(numeroFacesDetectadas)

    # Validação para verificar se existe apenas 1 face por imagem - Durante o treinamento o dlib requer apenas 1 face
    # caso contrário poderá ter problemas durante o treinamento
    if numeroFacesDetectadas > 1:
        print("Há mais de 1 face na imagem {}".format(arquivo))
        exit(0)
    elif numeroFacesDetectadas < 1:
        print("Nenhuma face encontrada no arquivo {}".format(arquivo))
        exit(0)

    # Percorremos cada um dos bounding boxes das faces detectadas
    for face in facesDetectadas:
        pontosFaciais = detectorPontos(imagem, face)
        descritorFacial = reconhecimentoFacial.compute_face_descriptor(imagem, pontosFaciais)

        # print(format(arquivo))
        # print(len(descritorFacial))
        # print(descritorFacial)

        # Percorrendo o vetor dlib e inserindo os valores em uma lista (Conversão de tipos de dados)
        listaDescritorFacial = [df for df in descritorFacial]
        # Convertendo a lista para o tipo array numpy
        npArrayDescritorFacial = np.asarray(listaDescritorFacial, dtype=np.float64)
        # print(npArrayDescritorFacial)

        # Criando uma nova dimensão no array numpy
        npArrayDescritorFacial = npArrayDescritorFacial[np.newaxis, :]
        # print(npArrayDescritorFacial)

        # A cada iteração é encontrado os 128 pontos da face e então iremos concatenar cada resultado em um array np
        if descritoresFaciais is None:
            descritoresFaciais = npArrayDescritorFacial
        else:
            # Concatenamos e inserimos no eixo 0 (linha) para que cada linha diferencie os pontos de cada face
            descritoresFaciais = np.concatenate((descritoresFaciais, npArrayDescritorFacial), axis=0)

    indice[idx] = arquivo
    idx += 1

    # cv2.imshow("Treinamento", imagem)
    # cv2.waitKey(0)

# print("Tamanho: {} - Formato: {}".format(len(descritoresFaciais), descritoresFaciais.shape))
# print(descritoresFaciais)
# print(indice)
np.save("imagens-e-recursos/recursos/descritores_yale.npy", descritoresFaciais)
# O parâmetro "f" abaixo significa de "file",
with open("imagens-e-recursos/recursos/indices_yales.pickle", "wb") as f:
    cPickle.dump(indice, f)

# cv2.destroyAllWindows()
