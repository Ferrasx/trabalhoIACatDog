# Importação das bibliotecas principais
import numpy as np  # Álgebra linear
import pandas as pd  # Processamento de dados, leitura de arquivos CSV, etc.
import os

# Importação das bibliotecas necessárias para a criação da Rede Neural
import matplotlib.pyplot as plt
import PIL
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

print(f"tensorflow version: {tf.__version__}")

# Caminho para o conjunto de dados
datasetPath = " PAHT (EXEMPLO PROFESSORA ) = https://www.kaggle.com/datasets/williamu32/dataset-bart-or-homer?resource=download"

# Caminhos para os conjuntos de treinamento e teste
treinamentoDatasetPath = os.path.join(datasetPath, "training_set")
testeDatasetPath = os.path.join(datasetPath, "test_set")

# Caminhos para as pastas contendo as imagens de cada classe no conjunto de treinamento
BartTreinamentoDatasetPath = os.path.join(treinamentoDatasetPath, "bart")
HomerTreinamentoDatasetPath = os.path.join(treinamentoDatasetPath, "homer")

# Caminhos para as pastas contendo as imagens de cada classe no conjunto de teste
BartTesteDatasetPath = os.path.join(testeDatasetPath, "bart")
HomerTesteDatasetPath = os.path.join(testeDatasetPath, "homer")

# Contagem do número de imagens em cada classe no conjunto de treinamento
qntdBartTreinamento = len(os.listdir(BartTreinamentoDatasetPath))
qntdHomerTreinamento = len(os.listdir(HomerTreinamentoDatasetPath))

# Contagem do número de imagens em cada classe no conjunto de teste
qntdBartTeste = len(os.listdir(BartTesteDatasetPath))
qntdHomerTeste = len(os.listdir(HomerTesteDatasetPath))

# Contagem total de imagens no conjunto de treinamento e teste
qntdTreinos = qntdBartTreinamento + qntdHomerTreinamento
qntdTeste = qntdBartTeste + qntdHomerTeste

print("Quantidade de Barts de Treinamento: ", qntdBartTreinamento)
print("Quantidade de Homers de Treinamento: ", qntdHomerTreinamento)
print("")
print("Quantidade de Barts de Teste: ", qntdBartTeste)
print("Quantidade de Homers de Teste: ", qntdHomerTeste)
print("")
print("Quantidade de imagens de treinamento: ", qntdTreinos)
print("Quantidade de imagens de teste: ", qntdTeste)

# Exemplo de visualização de uma imagem do conjunto de treinamento do Bart e do Homer
imgBart = os.path.join(
    BartTreinamentoDatasetPath, os.listdir(BartTreinamentoDatasetPath)[26]
)
PIL.Image.open(imgBart)
imgHomer = os.path.join(
    HomerTreinamentoDatasetPath, os.listdir(HomerTreinamentoDatasetPath)[26]
)
PIL.Image.open(imgHomer)

# Tamanho do lote (batch size) e tamanho da imagem
tamanhoLote = 32
tamanhoImagem = (160, 160)

# Criação dos geradores de dados para o conjunto de treinamento e teste com aumento de dados
treinoImageGen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=7,
    horizontal_flip=True,
    shear_range=0.2,
    height_shift_range=0.05,
    zoom_range=0.2,
)

testeImageGen = ImageDataGenerator(rescale=1.0 / 255)

treinoDataGen = treinoImageGen.flow_from_directory(
    batch_size=tamanhoLote,
    directory=treinamentoDatasetPath,
    shuffle=True,
    target_size=tamanhoImagem,
    class_mode="binary",
)

testeDataGen = testeImageGen.flow_from_directory(
    batch_size=tamanhoLote,
    directory=testeDatasetPath,
    shuffle=True,
    target_size=tamanhoImagem,
    class_mode="binary",
)

# Exemplo de visualização de algumas imagens geradas pelo gerador de dados de treinamento
umTesteSimples, _ = next(treinoDataGen)


def plotImages(imagesArray):
    fig, axes = plt.subplots(1, 5, figsize=(10, 10))
    axes = axes.flatten()
    for img, ax in zip(imagesArray, axes):
        ax.imshow(img)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


plotImages(umTesteSimples[:26])

# Criação do modelo da rede neural
model = keras.Sequential(
    [
        layers.Conv2D(
            16, 3, padding="same", activation="elu", input_shape=(160, 160, 3)
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="elu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="elu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(512, activation="elu"),
        layers.Dropout(0.2),
        layers.Dense(512, activation="elu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()


# Mostra o progresso do treinamento imprimindo um único ponto para cada epoch completada
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print("")
        print(".>>", end="")


steps_per_epoch = treinoDataGen.samples // treinoDataGen.batch_size
validation_steps = testeDataGen.samples // testeDataGen.batch_size
epochs = 100  # Quantidade de vezes que os dados serão mostrados para a rede neural

# Treinamento do modelo
history = model.fit(
    treinoDataGen,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=testeDataGen,
    validation_steps=validation_steps,
    callbacks=[PrintDot()],
    verbose=0,
)

# Avaliação do modelo
eval_results = model.evaluate(testeDataGen)
print("Acurácia: {:2.2%}".format(eval_results[1]))

# Exemplo de previsão usando uma imagem de teste do Homer
test_bart = os.path.join(
    BartTreinamentoDatasetPath, os.listdir(BartTreinamentoDatasetPath)[5]
)
homer_test = os.path.join(
    HomerTreinamentoDatasetPath, os.listdir(HomerTreinamentoDatasetPath)[13]
)
inv_map = {treinoDataGen.class_indices[k]: k for k in treinoDataGen.class_indices}

imagem_teste = image.load_img(homer_test, target_size=(160, 160))
imagem_teste = image.img_to_array(imagem_teste)
imagem_teste /= 255
imagem_teste = np.expand_dims(imagem_teste, axis=0)

previsao = model.predict(imagem_teste).flatten()
prev_name = tf.where(previsao < 0.5, 0, 1).numpy()

inv_map[prev_name[0]], previsao

# Exemplo de previsão usando a primeira imagem do conjunto de teste
primeiroTeste, nome = next(testeDataGen)
pred = model.predict(primeiroTeste).flatten()
pred = tf.where(pred < 0.5, 0, 1)
plt.imshow(primeiroTeste[0])
title = inv_map[pred.numpy()[0]]
plt.title(f"Nome: {title}")
plt.axis("off")
plt.show()
