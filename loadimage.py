import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np

# Load the trained AI model (make sure the model path is correct)
model = tf.keras.models.load_model('scratch_detector_model.h5')

# Create the main window
window = tk.Tk()
window.title("Scratch Detection AI")

# Function to open and load the image
def load_image():
    # Open the file dialog to select an image
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

    if file_path:
        # Open the image using PIL (Python Imaging Library)
        img = Image.open(file_path)
        
        # Resize the image to match the model's expected input size
        img = img.resize((224, 224))

        # Convert the image to an array and normalize it
        img_array = np.array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Display the image in the UI
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk

        # Call the AI model to predict if there is a scratch
        predict_image(img_array)

# Function to predict if the image has a scratch
def predict_image(img_array):
    # Get the AI model's prediction
    prediction = model.predict(img_array)
    print("prediction:", prediction)
    # If prediction > 0.5, the model thinks the image has a scratch
    if prediction >= 0.5:
        result_label.config(text="Scratchless", fg="green")
    else:
        result_label.config(text="Scratch detected", fg="red")

# UI Components
select_button = tk.Button(window, text="Select Image", command=load_image, font=("Arial", 14))
select_button.pack(pady=20)

image_label = tk.Label(window)
image_label.pack(pady=10)

result_label = tk.Label(window, text="", font=("Arial", 16))
result_label.pack(pady=10)

# Run the Tkinter event loop
window.mainloop()
