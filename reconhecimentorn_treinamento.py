import os
import glob
import _pickle as cPickle
import cv2
import dlib
import numpy as np

# Criação dos classificadores (detectores de face e pontos)
detectorFace = dlib.get_frontal_face_detector()
detectorPontos = dlib.shape_predictor("imagens-e-recursos/recursos/shape_predictor_68_face_landmarks.dat")
# Criação do classificador utilizando um modelo de reconhecimento facial já treinado
reconhecimentoFacial = dlib.face_recognition_model_v1("imagens-e-recursos/recursos/dlib_face_recognition_resnet_model_v1.dat")

indice = {}
idx = 0
descritoresFaciais = None

# Percorrendo a pasta de fotos para o treinamento
for arquivo in glob.glob(os.path.join("imagens-e-recursos/fotos/treinamento", "*.jpg")):
    imagem = cv2.imread(arquivo)
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
        # Armazenamento dos pontos faciais de cada face iterada das faces Detectadas
        # Passamos como parâmetro a imagem original (em cada iteração) e a face com as partes que interessa (ROI)
        pontosFaciais = detectorPontos(imagem, face)
        # Variável abaixo irá receber o que a CNN irá computar e encontrar as principais características da face
        # O resultado serão 128 posições que irão melhor descrever (características) a face que foi encontrada
        # O resultado da variável abaixo é propriamente dito o treinamento gerado pelas redes neurais convulocionais
        # O objetivo do CNN é encontrar as principais/melhores características (features).
        # O resultado de 128 posições são as principais características, de por exemplo, uma imagem de 700x600 pixels
        # que irá dar um valor de 420000 pixels. Ou seja, reduzindo as entradas da CNN de forma drástica.
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
np.save("imagens-e-recursos/recursos/descritores_rn.npy", descritoresFaciais)
# O parâmetro "f" abaixo significa de "file",
with open("imagens-e-recursos/recursos/indices_rn.pickle", "wb") as f:
    cPickle.dump(indice, f)

# cv2.destroyAllWindows()
