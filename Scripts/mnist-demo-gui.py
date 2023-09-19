import tkinter as tk
from PIL import ImageGrab
import numpy as np
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf


def paint(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black", width=25)
    
def canvas_to_array():
    # Capture the canvas content as an image
    canvas.update()
    x0 = canvas.winfo_rootx()
    y0 = canvas.winfo_rooty()
    x1 = x0 + canvas.winfo_width()
    y1 = y0 + canvas.winfo_height()
    image = ImageGrab.grab(bbox=(x0, y0, x1, y1))
    
    # Convert the image to a NumPy array
    img_array = np.array(image)
    img = cv2.resize(img_array, (28, 28), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = 255 - img
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = img / 225
    img = img.reshape(1, -1)
    
    return img
    
def predict(img):
    predictions = model.predict(img)
    predicted_class = tf.argmax(predictions, axis=-1)
    prediction_label.config(text=f'The predicted number is {predicted_class[0]}')


model = tf.keras.models.load_model('Models/MNIST-Classifier-Model.h5')

root = tk.Tk()
root.title("Drawable Canvas")
root.resizable(False, False)

frame = tk.Frame(root, width=200, height=100, bd=2, relief=tk.SUNKEN)
frame.pack(side=tk.TOP)

predict_button = tk.Button(frame, text="Predict", command=lambda: predict(canvas_to_array()))
predict_button.pack(side=tk.LEFT)

prediction_label = tk.Label(frame, text='No Predictions Yet')
prediction_label.pack()

canvas = tk.Canvas(root, bg="white", width=400, height=400)
canvas.pack(side=tk.TOP)
    
clear_button = tk.Button(root, text="Clear Canvas", command=lambda: canvas.delete("all"))
clear_button.pack(side=tk.BOTTOM)

canvas.bind("<B1-Motion>", paint)

root.mainloop()