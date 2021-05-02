import os
import glob
import dlib
import cv2


detectorRelogio = dlib.simple_object_detector("imagens-e-recursos/recursos/detector_relogios.svm")
detectorPontosRelogio = dlib.shape_predictor("imagens-e-recursos/recursos/detector_relogios_pontos.dat")

# Exibe um valor de forma similar aos valores de confiabilidade que vimos em outras aulas, quanto maior o valor melhor
print(dlib.test_shape_predictor("imagens-e-recursos/recursos/teste_relogios_pontos.xml",
                                "imagens-e-recursos/recursos/detector_relogios_pontos.dat"))


def imprimePontos(imagem, pontos):
    for p in pontos.parts():
        cv2.circle(imagem, (p.x, p.y), 2, (0, 255, 0))


for arquivo in glob.glob(os.path.join("imagens-e-recursos/relogios_teste", "*.jpg")):
    imagem = cv2.imread(arquivo)

    objetosDetectados = detectorRelogio(imagem, 2)
    for relogio in objetosDetectados:
        l, t, r, b = (int(relogio.left()), int(relogio.top()), int(relogio.right()), int(relogio.bottom()))
        cv2.rectangle(imagem, (l, t), (r, b), (0, 0, 255), 2)
        pontos = detectorPontosRelogio(imagem, relogio)
        imprimePontos(imagem, pontos)

    cv2.imshow("Detector pontos", imagem)
    cv2.waitKey(0)

cv2.destroyAllWindows()
