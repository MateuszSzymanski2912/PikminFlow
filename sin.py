import numpy as np
import matplotlib.pyplot as plt
from layers import Dense
from model import Model
from matplotlib.animation import FuncAnimation, FFMpegWriter


X = np.linspace(0, 4*np.pi, 25).reshape(-1, 1)
y = np.sin(X)
X_scaled = (X - 2*np.pi) / (2*np.pi)  # Normalize to [-1, 1]

layers = [Dense(input_size=1, dim = 64, activation='tanh'),
            Dense(dim = 64, activation = 'tanh'),
            Dense(dim = 1, activation='linear')]

model = Model()
model.initialize(layers=layers, batch_size = 32, epochs = 15000, learning_rate=0.01, optimizer='adam', loss_function='mse')


x_plot = np.linspace(0, 4*np.pi, 200).reshape(-1, 1)
y_true = np.sin(x_plot)
x_plot_scaled = (x_plot - 2*np.pi) / (2*np.pi)  # Normalize to [-1, 1]
model.train_with_animation(X_scaled, y, k = 75, x_plot=x_plot_scaled)

fig, axes = plt.subplots(2, 1, figsize=(10, 6))
line_pred, = axes[0].plot([], [], 'r-', label='NN prediction')
line_true, = axes[0].plot(x_plot, y_true, 'g-', label='True sin(x)')
axes[0].set_ylim(-1.2, 1.2)
title = axes[0].set_title(f"Visualization of neural network teaching itself sin function - Epoch {model.prediction_epochs[0]}")
axes[0].legend()
axes[0].grid(True)

line_loss, = axes[1].plot([], [], 'b-', label = 'Loss')
axes[1].set_xlim(0, model.epochs)
axes[1].set_ylim(0, 1.1*max(model.history['loss']))
axes[1].legend()
axes[1].grid(True)

def init():
    line_pred.set_data([], [])
    line_loss.set_data([], [])
    return line_pred, line_loss,

def update(frame):
    y_pred = model.predictions_history[frame]
    line_pred.set_data(x_plot, y_pred)
    line_loss.set_data(model.epoch_indices[:model.k*(frame + 1)], model.history['loss'][:model.k*(frame + 1)])
    title.set_text(f'Visualization of neural network teaching itself sin function - Epoch {model.prediction_epochs[frame]}')
    return line_pred, line_loss, title

ani = FuncAnimation(fig, update, frames=len(model.predictions_history),
                    init_func=init, blit=False, interval=200)

FFwriter = FFMpegWriter(fps=60)
ani.save("visualization_sin.gif", writer=FFwriter)
plt.show()
