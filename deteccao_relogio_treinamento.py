import dlib

# Variável abaixo está armazenando o objeto da biblioteca dlib
opcoes = dlib.simple_object_detector_training_options()
# O parâmetro TRUE abaixo é muito utilizada caso você esteja treinando o seu próprio detector facial
# No nosso caso atual, é como se "inclinássemos" a imagem do relógio um pouco para o lado para melhorar a detecção.
opcoes.add_left_right_image_flips = True
# O parâmetro C abaixo é o custo do algoritmo SVM
opcoes.C = 5

# Estamos realizando o treinamento do nosso algoritmo, onde passamos os parâmetros de:
# primeiramente as imagens com os bounding boxes que fizemos no imglab e depois o nome do nosso algoritmo com extensão.
dlib.train_simple_object_detector("imagens-e-recursos/recursos/treinamento_relogios.xml",
                                  "imagens-e-recursos/recursos/detector_relogios.svm", opcoes)

# NOTA: Caso dê erro de "Unable to open file: <imagens>
# Isto acontece porque não está sendo possível encontrar as imagens que o XML está apontando. Para corrigir, basta
# abrir o XML e indicar o diretório correto das imagens que estão no XML
