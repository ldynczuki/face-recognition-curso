import dlib

opcoes = dlib.shape_predictor_training_options()

# Geração do preditor de formas, onde utilizamos o xml que nós incluímos os 8 pontos nas imagens dos relógios
dlib.train_shape_predictor("imagens-e-recursos/recursos/treinamento_relogios_pontos.xml",
                           "detector_relogios_pontos.dat", opcoes)