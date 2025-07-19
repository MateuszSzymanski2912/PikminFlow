import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist
from layers import *
from model import Model
from matplotlib.animation import FuncAnimation, FFMpegWriter


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0], -1)) / 255.0
    x_test = x_test.reshape((x_test.shape[0], -1)) / 255.0
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_mnist()
X = x_train[:100]
y = y_train[:100]
x_plot = x_train[0].reshape(1, -1)

layers = [Dense(input_size=28*28, dim = 512, activation='relu'),
          Dropout(p=0.1),
          Dense(dim = 28*28, activation = 'linear')]

model = Model()
model.initialize(layers=layers, batch_size = 32, epochs = 1000, learning_rate=0.005, optimizer='adam', loss_function='mse')
model.train_with_animation(X, X, x_plot, k = 5)


fig, axes = plt.subplot_mosaic(
    """
    AB
    CC
    """,
    figsize=(10,6)
)
im_pred = axes['A'].imshow(x_plot.reshape(28, 28), aspect='auto', cmap='gray')
im_true = axes['B'].imshow(x_plot.reshape(28, 28), aspect='auto', cmap='gray')
line_loss, = axes['C'].plot([], [], 'b-', label = 'Loss')
axes['C'].set_xlim(0, model.epochs)
axes['C'].set_ylim(0, 1.1*max(model.history['train_loss']))
axes['C'].legend()
axes['C'].grid(True)

title = fig.suptitle(f"Visualization of autoencoder teaching itself MNIST images \n Epoch {model.prediction_epochs[0]}")
axes['A'].set_title("AE prediction")
axes['B'].set_title("True image")

def init():
    im_pred.set_data(np.zeros((28, 28)))
    line_loss.set_data([], [])
    return im_pred, line_loss,

def update(frame):
    y_pred = model.predictions_history[frame]
    im_pred.set_data(y_pred.reshape(28, 28))
    line_loss.set_data(model.epoch_indices[:model.k*(frame + 1)], model.history['train_loss'][:model.k*(frame + 1)])
    title.set_text(f"Visualization of autoencoder teaching itself MNIST images \n Epoch {model.prediction_epochs[frame]}")
    return im_pred, line_loss, title

ani = FuncAnimation(fig, update, frames=len(model.predictions_history),
                    init_func=init, blit=False, interval=200)

FFwriter = FFMpegWriter(fps=60)
ani.save("visualization_mnist.gif", writer=FFwriter)
plt.show()