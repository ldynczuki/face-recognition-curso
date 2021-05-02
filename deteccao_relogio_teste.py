import os
import dlib
import cv2
import glob

# Exibe os valores de precisão de acertos das imagens dos relogios que manualmente criamos os bounding boxes e o svm.
print(dlib.test_simple_object_detector("imagens-e-recursos/recursos/teste_relogios.xml",
                                       "imagens-e-recursos/recursos/detector_relogios.svm"))

# Variável utilizada para receber o objeto classificador que criamos durante o treinamento
detectorRelogio = dlib.simple_object_detector("imagens-e-recursos/recursos/detector_relogios.svm")

for imagem in glob.glob(os.path.join("imagens-e-recursos/relogios_teste", "*.jpg")):
    img = cv2.imread(imagem)
    objetosDetectados = detectorRelogio(img, 2)

    for d in objetosDetectados:
        l, t, r, b = (int(d.left()), int(d.top()), int(d.right()), int(d.bottom()))
        cv2.rectangle(img, (l, t), (r, b), (0, 0, 255), 2)

    cv2.imshow("Detector Relogios", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
