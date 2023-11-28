import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import customtkinter
from io import BytesIO
import numpy as np
import cv2

width, height = 400, 400

# Setting Custom Appearance for the application
customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

# Size of the window
root = customtkinter.CTk()
root.geometry("900x600")

def np_im_to_data(im):
    array = np.array(im, dtype=np.uint8)
    im = Image.fromarray(array)
    with BytesIO() as output:
        im.save(output, format='PNG')
        data = output.getvalue()
    return data

def on_contrast_slider_move(value):
    print("Test")


def load_image(max_size=(400,400)):
    file_path = filedialog.askopenfilename(title="Pick an Image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        image = cv2.imread(file_path)
        if image is not None:
            img_height, img_width = image.shape[:2]
            max_height, max_width = max_size
            if img_height > max_height or img_width > max_width:
                scale = min(max_height / img_height, max_width / img_width)
                new_height = int(img_height * scale)
                new_width = int(img_width * scale)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_image = ImageTk.PhotoImage(image=Image.fromarray(image)) 
            image_panel.create_image(0, 0, anchor="nw", image=original_image)



#Creates the Film Effects Window
def film_effects_ui():
    top = customtkinter.CTkToplevel()
    top.geometry("500x400")

    frame_1 = customtkinter.CTkFrame(master=top)
    frame_2 = customtkinter.CTkFrame(master=top)
    frame_1.pack(pady=10, padx=10, fill="both", expand=True)
    frame_2.pack(pady=10, padx=10, fill="both", expand=True)

    label_1 = customtkinter.CTkLabel(frame_1, text="Preset Film Effects")
    btn_pre_1 = customtkinter.CTkButton(master=frame_1, text="Classic Vintage")
    btn_pre_2 = customtkinter.CTkButton(master=frame_1, text="Black and White")
    btn_pre_3 = customtkinter.CTkButton(master=frame_1, text="Painted")

    label_1.pack()
    btn_pre_1.pack(pady=10, padx=10, side=tk.LEFT)
    btn_pre_2.pack(pady=10, padx=10, side=tk.LEFT)
    btn_pre_3.pack(pady=10, padx=10, side=tk.LEFT)

    label_2 = customtkinter.CTkLabel(frame_2, text="Custom Effects")
    label_3 = customtkinter.CTkLabel(frame_2, text="Contrast")
    label_4 = customtkinter.CTkLabel(frame_2, text="Saturation")
    label_5 = customtkinter.CTkLabel(frame_2, text="Temperature")
    contrast = customtkinter.CTkSlider(frame_2, from_=0, to=100, command=on_contrast_slider_move)
    saturation = customtkinter.CTkSlider(frame_2, from_=0, to=100)
    temperature = customtkinter.CTkSlider(frame_2, from_=0, to=100)

    label_2.pack()
    label_3.pack()
    contrast.pack()
    label_4.pack()
    saturation.pack()
    label_5.pack()
    temperature.pack()

#Creates the Filters Window
def filters_ui():
    top = customtkinter.CTkToplevel()
    top.geometry("300x200")

    frame_1 = customtkinter.CTkFrame(master=top)
    frame_1.pack(pady=10, padx=10, fill="both", expand=True)

    label_1 = customtkinter.CTkLabel(frame_1, text="GrayScale")
    label_2 = customtkinter.CTkLabel(frame_1, text="Noise Reduction")
    label_3 = customtkinter.CTkLabel(frame_1, text="Light Leak")
    label_4 = customtkinter.CTkLabel(frame_1, text="Gamma Correction")

    grayscale = customtkinter.CTkSlider(frame_1, from_=0, to=100)
    noise = customtkinter.CTkSlider(frame_1, from_=0, to=100)
    light_light = customtkinter.CTkSlider(frame_1, from_=0, to=100)
    gamma = customtkinter.CTkSlider(frame_1, from_=0, to=100)

    label_1.pack()
    grayscale.pack()
    label_2.pack()
    noise.pack()
    label_3.pack()
    light_light.pack()
    label_4.pack()
    gamma.pack()

#Creates the Zoom Window
def zoom_ui():
    top = customtkinter.CTkToplevel()
    top.geometry("300x200")

    frame_1 = customtkinter.CTkFrame(master=top)
    frame_1.pack(pady=10, padx=10, fill="both", expand=True)
    label_1 = customtkinter.CTkLabel(frame_1, text="Zoom Level")
    zoom = customtkinter.CTkSlider(frame_1, from_=0, to=100)

    label_1.pack()
    zoom.pack()

#Creates the White Balance Window
def white_balance_ui():
    top = customtkinter.CTkToplevel()
    top.geometry("300x200")

    frame_1 = customtkinter.CTkFrame(master=top)
    frame_1.pack(pady=10, padx=10, fill="both", expand=True)
    label_1 = customtkinter.CTkLabel(frame_1, text="White Balance")
    white_balace = customtkinter.CTkSlider(frame_1, from_=0, to=100)

    label_1.pack()
    white_balace.pack()

#Creates the Tone Curve Window
def tone_curve_ui():
    top = customtkinter.CTkToplevel()
    top.geometry("300x200")

    frame_1 = customtkinter.CTkFrame(master=top)
    frame_1.pack(pady=10, padx=10, fill="both", expand=True)
    label_1 = customtkinter.CTkLabel(frame_1, text="Tone Curve")
    tone_curve = customtkinter.CTkSlider(frame_1, from_=0, to=100)

    label_1.pack()
    tone_curve.pack()


frame_1 = customtkinter.CTkFrame(master=root)
frame_2 = customtkinter.CTkFrame(master=root)
frame_3 = customtkinter.CTkFrame(master=root)

frame_1.pack(pady=10, padx=10, fill="both", expand=True)
frame_2.pack(pady=10, padx=10, fill="both", expand=True)
frame_3.pack(pady=10, padx=10, fill="both", expand=True)

image_panel = customtkinter.CTkCanvas(master=frame_1, width=width, height=height, bg='white')
image_panel.pack(pady=30, padx=10)

# Second Frame Buttons
save_button = customtkinter.CTkButton(frame_2, text="Save Image")
load_button = customtkinter.CTkButton(frame_2, text="Load Image", command=load_image)
quit_button = customtkinter.CTkButton(frame_2, text="Quit", command=root.quit)
reset_button = customtkinter.CTkButton(frame_2, text="Reset Image")

save_button.pack(side=tk.LEFT, pady=10, padx=20)
load_button.pack(side=tk.LEFT, pady=10, padx=20)
quit_button.pack(side=tk.RIGHT, pady=10, padx=20)
reset_button.pack(side=tk.RIGHT, pady=10, padx=20)

# Third Frame Buttons
film_effects_button = customtkinter.CTkButton(frame_3, text="Film Effects", command=film_effects_ui)
filters_button = customtkinter.CTkButton(frame_3, text="Filters", command=filters_ui)
zoom_button = customtkinter.CTkButton(frame_3, text="Zoom", command=zoom_ui)
white_balance_button = customtkinter.CTkButton(frame_3, text="White Balance", command=white_balance_ui)
tone_curve_button = customtkinter.CTkButton(frame_3, text="Tone Curve", command=tone_curve_ui)

film_effects_button.pack(side=tk.LEFT, pady=10, padx=20)
filters_button.pack(side=tk.LEFT, pady=10, padx=20)
zoom_button.pack(side=tk.LEFT, pady=10, padx=20)
white_balance_button.pack(side=tk.LEFT, pady=10, padx=20)
tone_curve_button.pack(side=tk.LEFT, pady=10, padx=20)

root.mainloop()
