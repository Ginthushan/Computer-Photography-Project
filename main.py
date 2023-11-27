import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import customtkinter
from io import BytesIO

width, height = 400, 400

#Setting Custom Appearance for the application
customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

#Size of the window
root = customtkinter.CTk()
root.geometry("900x600")

def np_im_to_data(im):
    array = np.array(im, dtype=np.uint8)
    im = Image.fromarray(array)
    with BytesIO() as output:
        im.save(output, format='PNG')
        data = output.getvalue()
    return data

def temp_function():
    ...


frame_1 = customtkinter.CTkFrame(master=root)
frame_2 = customtkinter.CTkFrame(master=root)
frame_3 = customtkinter.CTkFrame(master=root)

frame_1.pack(pady= 10, padx= 10, fill="both", expand=True)
frame_2.pack(pady= 10, padx= 10, fill="both", expand=True)
frame_3.pack(pady= 10, padx= 10, fill="both", expand=True)

image_panel = customtkinter.CTkCanvas(master = frame_1, width=width, height=height, bg='white')
image_panel.pack(pady= 30, padx= 10)

#Second Frame Buttons
save_button = customtkinter.CTkButton(frame_2, text="Save Image", command=temp_function)
load_button = customtkinter.CTkButton(frame_2, text="Load Image", command=temp_function)
quit_button = customtkinter.CTkButton(frame_2, text="Quit", command=root.quit)
reset_button = customtkinter.CTkButton(frame_2, text="Reset Image", command=temp_function)

save_button.pack(side=tk.LEFT, pady= 10, padx= 20)
load_button.pack(side=tk.LEFT, pady= 10, padx= 20)
quit_button.pack(side=tk.RIGHT, pady= 10, padx= 20)
reset_button.pack(side=tk.RIGHT, pady= 10, padx= 20)

#Third Frame Buttons
film_effects_button = customtkinter.CTkButton(frame_3, text="Film Effects", command=temp_function)
filters_button = customtkinter.CTkButton(frame_3, text="Filters", command=temp_function)
zoom_buttom = customtkinter.CTkButton(frame_3, text="Zoom", command=temp_function)
white_balance_buttom = customtkinter.CTkButton(frame_3, text="White Balance", command=temp_function)
tone_curve_buttom = customtkinter.CTkButton(frame_3, text="Tone Curve", command=temp_function)

film_effects_button.pack(side=tk.LEFT, pady= 10, padx= 20)
filters_button.pack(side=tk.LEFT, pady= 10, padx= 20)
zoom_buttom.pack(side=tk.LEFT, pady= 10, padx= 20)
white_balance_buttom.pack(side=tk.LEFT, pady= 10, padx= 20)
tone_curve_buttom.pack(side=tk.LEFT, pady= 10, padx= 20)


root.mainloop()