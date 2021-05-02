import os
import glob
import _pickle as cPickle
import dlib
import cv2
import numpy as np

detectorFace = dlib.get_frontal_face_detector()
detectorPontos = dlib.shape_predictor("imagens-e-recursos/recursos/shape_predictor_68_face_landmarks.dat")
reconhecimentoFacial = dlib.face_recognition_model_v1("imagens-e-recursos/recursos/dlib_face_recognition_resnet_model_v1.dat")
indices = np.load("imagens-e-recursos/recursos/indices_rn.pickle", allow_pickle=True)
descritoresFaciais = np.load("imagens-e-recursos/recursos/descritores_rn.npy")
limiar = 0.5

for arquivo in glob.glob(os.path.join("imagens-e-recursos/fotos", "*.jpg")):
    imagem = cv2.imread(arquivo)
    facesDetectadas = detectorFace(imagem, 2)
    for face in facesDetectadas:
        l, t, r, b = (int(face.left()), int(face.top()), int(face.right()), int(face.bottom()))

        pontosFaciais = detectorPontos(imagem, face)
        descritorFacial = reconhecimentoFacial.compute_face_descriptor(imagem, pontosFaciais)
        listaDescritorFacial = [df for df in descritorFacial]
        npArrayDescritorFacial = np.asarray(listaDescritorFacial, dtype=np.float64)
        npArrayDescritorFacial = npArrayDescritorFacial[np.newaxis, :]

        # A variável irá armazenar o resultado da distância euclidiana (KNN) comparado a cada imagem de treinamento
        # O resultado exibido a seguir terá uma quantidade igual a quantidade de imagens no treinamento
        distancias = np.linalg.norm(npArrayDescritorFacial - descritoresFaciais, axis=1)
        # print("Distâncias: {}".format(distancias))

        minimo = np.argmin(distancias)
        # print(minimo)
        distanciaMinima = distancias[minimo]
        # print(distanciaMinima)

        if distanciaMinima <= limiar:
            nome = os.path.split(indices[minimo])[1].split(".")[0]
        else:
            nome = ' '

        cv2.rectangle(imagem, (l, t), (r, b), (0, 255, 255), 2)
        texto = "{} {: 4f}".format(nome, distanciaMinima)
        cv2.putText(imagem, texto, (r, t), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 255))

    cv2.imshow("Reconhecimento Facial", imagem)
    cv2.waitKey(0)

cv2.destroyAllWindows()
