import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageEnhance
import customtkinter
from io import BytesIO
import numpy as np
import cv2
from numpy import asarray

width, height = 400, 400

global original_image_data
global reset_image_data

# Setting Custom Appearance for the application
customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

# Size of the window
root = customtkinter.CTk()
root.title("Computer Photography Project")
root.geometry("1000x600")


def on_contrast_slider_move(value):

    image_data = original_image_data

    image_data = cv2.addWeighted(image_data, 1 + value / 100, np.zeros_like(image_data), 0, 0)
    res = ImageTk.PhotoImage(image=Image.fromarray(image_data))
    image_panel.image = res
    image_panel.create_image(0, 0, anchor="nw", image=res)

def on_saturation_slider_move(value):

    pil_image = Image.fromarray(original_image_data)
    enhancer = ImageEnhance.Color(pil_image)
    saturated_image = enhancer.enhance(1 + value / 100)
    np_image_data = ImageTk.PhotoImage(image=saturated_image)
    image_panel.image = np_image_data
    image_panel.create_image(0, 0, anchor="nw", image=np_image_data)

def on_temperature_slider_move(value):
    global original_image_data
    image_data = original_image_data.astype(float)
    temperature_factor = value / 100

    color_matrix = np.array([
        [1, 0, 0],               # Red channel
        [0, 1 - temperature_factor, 0],  # Green channel
        [0, 0, 1 + temperature_factor]   # Blue channel
    ])

    adjusted_image_data = np.dot(image_data, color_matrix.T)
    adjusted_image_data = np.clip(adjusted_image_data, 0, 255)
    adjusted_image_data = adjusted_image_data.astype(np.uint8)

    np_image_data = ImageTk.PhotoImage(image=Image.fromarray(adjusted_image_data))
    image_panel.image = np_image_data
    image_panel.create_image(0, 0, anchor="nw", image=np_image_data)

def load_image(max_size=(400,400)):
    global original_image_data
    global reset_image_data
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
            original_image_data = image
            original_image_data = asarray(original_image_data)
            reset_image_data = np.copy(original_image_data)
            original_image = ImageTk.PhotoImage(image=Image.fromarray(image))
            image_panel.image = original_image
            image_panel.create_image(0, 0, anchor="nw", image=original_image)

def save_image():
    global original_image_data
    image_data = cv2.cvtColor(original_image_data, cv2.COLOR_RGB2BGR)
    cv2.imwrite('modified_image.jpg', image_data)

def reset_image():
    global original_image_data
    image_data = original_image_data.astype(np.uint8)
    np_image_data = ImageTk.PhotoImage(image=Image.fromarray(image_data))
    image_panel.image = np_image_data
    image_panel.create_image(0, 0, anchor="nw", image=np_image_data)

def classic_vintage(contrast_factor=1.5, saturation_factor=0.8, noise_factor=5):
    global original_image_data

    img_array = original_image_data.astype(float)

    # Adjust contrast and saturation for each color channel
    for i in range(3):  # Loop through RGB channels
        img_array[:, :, i] = img_array[:, :, i] * contrast_factor
        img_array[:, :, i] = np.clip(img_array[:, :, i], 0, 255)
        img_array[:, :, i] = img_array[:, :, i].astype(np.uint8)

    # Convert to grayscale
    grayscale = np.dot(img_array[..., :3], [0.299, 0.587, 0.114])

    # Add noise
    noise = np.random.normal(scale=noise_factor, size=grayscale.shape).astype(np.uint8)
    img_array = np.clip(grayscale + noise, 0, 255)

    # Adjust saturation
    img_array = img_array * saturation_factor
    vintage_image = np.clip(img_array, 0, 255)
    vintage_image = vintage_image.astype(np.uint8)

    # Update the image on the canvas
    np_image_data = ImageTk.PhotoImage(image=Image.fromarray(vintage_image))
    image_panel.image = np_image_data
    image_panel.create_image(0, 0, anchor="nw", image=np_image_data)

def black_white():
    global original_image_data

    image_data = original_image_data.astype(float)
    weights = np.array([0.2989, 0.587, 0.114])
    grayscale_image = np.dot(image_data, weights)
    grayscale_image = np.clip(grayscale_image, 0, 255)

    #original_image_data = grayscale_image.astype(np.uint8)
    grayscale_image = grayscale_image.astype(np.uint8)

    np_image_data = ImageTk.PhotoImage(image=Image.fromarray(grayscale_image))
    image_panel.image = np_image_data
    image_panel.create_image(0, 0, anchor="nw", image=np_image_data)

def draw_strokes(stroke_width_range=(1, 2), stroke_length_range=(1, 2)):
    global original_image_data

    image_data = original_image_data

    height, width, _ = image_data.shape
    sampled_pixels = np.random.choice(np.arange(height * width), size=(height * width) // 10, replace=False)

    for pixel in sampled_pixels:
        y, x = divmod(pixel, width)
        y = np.clip(y, 0, height - 1)
        x = np.clip(x, 0, width - 1)

        stroke_width = np.random.randint(stroke_width_range[0], stroke_width_range[1] + 1)
        stroke_length = np.random.randint(stroke_length_range[0], stroke_length_range[1] + 1)

        color = tuple(map(int, image_data[y, x]))

        end_x = min(x + stroke_length, width - 1)
        end_y = min(y + stroke_width, height - 1)

        cv2.line(image_data, (x, y), (end_x, end_y), color, stroke_width)

    np_image_data = ImageTk.PhotoImage(image=Image.fromarray(image_data))
    image_panel.image = np_image_data
    image_panel.create_image(0, 0, anchor="nw", image=np_image_data)

def on_grayscale_slider_move(value):
    global original_image_data

    image_data = original_image_data.astype(float)
    grayscale_factor = value / 100
    grayscale_image = np.dot(image_data, [grayscale_factor, 1 - grayscale_factor, 1 - grayscale_factor])
    grayscale_image = np.clip(grayscale_image, 0, 255)
    grayscale_image = grayscale_image.astype(np.uint8)

    np_image_data = ImageTk.PhotoImage(image=Image.fromarray(grayscale_image))
    image_panel.image = np_image_data
    image_panel.create_image(0, 0, anchor="nw", image=np_image_data)

def on_noise_slider_move(value):
    global original_image_data

    image_data = original_image_data.astype(float)
    noise_level = value * 2.55
    noise = np.random.normal(loc=0, scale=noise_level, size=image_data.shape)
    noisy_image = image_data + noise
    noisy_image = np.clip(noisy_image, 0, 255)
    noisy_image = noisy_image.astype(np.uint8)


    np_image_data = ImageTk.PhotoImage(image=Image.fromarray(noisy_image))
    image_panel.image = np_image_data
    image_panel.create_image(0, 0, anchor="nw", image=np_image_data)

def on_light_leak_slider_move():
    global original_image_data

    image_data = original_image_data.astype(float)
    new_layer = original_image_data.astype(float)

    # Choose a gradient (radial gradient for a more pronounced effect)
    height, width, _ = image_data.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    center_x, center_y = width // 2, height // 2
    radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    gradient = 1 - np.clip(radius / (width / 2), 0, 1)

    # Adjust Gradient Settings (apply colors to the gradient)
    rainbow_colors = np.array([
    [148, 0, 211],  # Violet
    [75, 0, 130],   # Indigo
    [0, 0, 255],    # Blue
    [0, 255, 0],    # Green
    [255, 255, 0],  # Yellow
    [255, 127, 0],  # Orange
    [255, 0, 0]     # Red
])
    gradient_colored = gradient[:, :, np.newaxis] * rainbow_colors[np.newaxis, 0, :]

    # Draw the Gradient
    new_layer *= (1 - gradient[:, :, np.newaxis])  # Make the original image darker
    new_layer += gradient_colored  # Add the rainbow-colored gradient


    # Blend Mode: Screen
    brightened_image = np.clip(image_data + new_layer, 0, 255)
    brightened_image = brightened_image.astype(np.uint8)

    np_image_data = ImageTk.PhotoImage(image=Image.fromarray(brightened_image))
    image_panel.image = np_image_data
    image_panel.create_image(0, 0, anchor="nw", image=np_image_data)

def on_gamma_slider_move(value):
    global original_image_data

    image_data = original_image_data.astype(float)
    gamma_value = value / 100
    gamma_corrected_image = 255.0 * (image_data / 255.0) ** (1 / gamma_value)
    gamma_corrected_image = np.clip(gamma_corrected_image, 0, 255)

    gamma_corrected_image = gamma_corrected_image.astype(np.uint8)

    np_image_data = ImageTk.PhotoImage(image=Image.fromarray(gamma_corrected_image))
    image_panel.image = np_image_data
    image_panel.create_image(0, 0, anchor="nw", image=np_image_data)

def white_balance_grey():
    mean_r, mean_g, mean_b = np.average(original_image_data.reshape(-1,3),0)
    mean_gray= 128
    scale_r = mean_gray / mean_r
    scale_g = mean_gray / mean_g
    scale_b = mean_gray / mean_b

    result_image = np.empty(original_image_data.shape, dtype=np.uint8)
    result_image[:,:,0] = np.clip(original_image_data[:,:,0] * scale_r, 0, 255).astype(np.uint8)
    result_image[:,:,1] = np.clip(original_image_data[:,:,1] * scale_g, 0, 255).astype(np.uint8)
    result_image[:,:,2] = np.clip(original_image_data[:,:,2] * scale_b, 0, 255).astype(np.uint8)

    np_image_data = ImageTk.PhotoImage(image=Image.fromarray(result_image))
    image_panel.image = np_image_data
    image_panel.create_image(0, 0, anchor="nw", image=np_image_data)

def white_balance_white():
    mean_r, mean_g, mean_b = np.average(original_image_data.reshape(-1,3),0)
    mean_white=255
    scale_r = mean_white / mean_r
    scale_g = mean_white / mean_g
    scale_b = mean_white / mean_b

    result_image = np.empty(original_image_data.shape, dtype=np.uint8)
    result_image[:,:,0] = np.clip(original_image_data[:,:,0] * scale_r, 0, 255).astype(np.uint8)
    result_image[:,:,1] = np.clip(original_image_data[:,:,1] * scale_g, 0, 255).astype(np.uint8)
    result_image[:,:,2] = np.clip(original_image_data[:,:,2] * scale_b, 0, 255).astype(np.uint8)

    np_image_data = ImageTk.PhotoImage(image=Image.fromarray(result_image))
    image_panel.image = np_image_data
    image_panel.create_image(0, 0, anchor="nw", image=np_image_data)

def on_zoom_slider_move(value):
    global original_image_data

    image_data = original_image_data.astype(float)

    zoom_level = 1 + (value / 100)

    zoomed_image = cv2.resize(image_data, None, fx=zoom_level, fy=zoom_level, interpolation=cv2.INTER_LINEAR)
    zoomed_image = np.clip(zoomed_image, 0, 255)

    zoomed_image = zoomed_image.astype(np.uint8)

    np_image_data = ImageTk.PhotoImage(image=Image.fromarray(zoomed_image))
    image_panel.image = np_image_data
    image_panel.create_image(0, 0, anchor="nw", image=np_image_data)

def on_white_balance_slider_move(value):
    global original_image_data

    image_data = original_image_data.astype(float)
    white_balance_factor = 1 + (value / 100)
    white_balanced_image = image_data * white_balance_factor
    white_balanced_image = np.clip(white_balanced_image, 0, 255)

    white_balanced_image  = white_balanced_image.astype(np.uint8)
    np_image_data = ImageTk.PhotoImage(image=Image.fromarray(white_balanced_image ))
    image_panel.image = np_image_data
    image_panel.create_image(0, 0, anchor="nw", image=np_image_data)

def on_tone_curve_slider_move(value):
    global original_image_data

    image_data = original_image_data.astype(float)
    tone_curve_factor = 1 + (value / 100)
    toned_image = 255 * (image_data / 255) ** tone_curve_factor

    toned_image = np.clip(toned_image, 0, 255)
    toned_image = toned_image.astype(np.uint8)

    np_image_data = ImageTk.PhotoImage(image=Image.fromarray(toned_image))
    image_panel.image = np_image_data
    image_panel.create_image(0, 0, anchor="nw", image=np_image_data)


#Creates the Film Effects Window
def film_effects_ui():
    top = customtkinter.CTkToplevel()
    top.geometry("500x400")
    top.title("Film Effects")

    frame_1 = customtkinter.CTkFrame(master=top)
    frame_2 = customtkinter.CTkFrame(master=top)
    frame_1.pack(pady=10, padx=10, fill="both", expand=True)
    frame_2.pack(pady=10, padx=10, fill="both", expand=True)

    label_1 = customtkinter.CTkLabel(frame_1, text="Preset Film Effects")
    btn_pre_1 = customtkinter.CTkButton(master=frame_1, text="Classic Vintage", command=classic_vintage)
    btn_pre_2 = customtkinter.CTkButton(master=frame_1, text="Black and White", command=black_white)
    btn_pre_3 = customtkinter.CTkButton(master=frame_1, text="Painted", command=draw_strokes)

    label_1.pack()
    btn_pre_1.pack(pady=10, padx=10, side=tk.LEFT)
    btn_pre_2.pack(pady=10, padx=10, side=tk.LEFT)
    btn_pre_3.pack(pady=10, padx=10, side=tk.LEFT)

    label_2 = customtkinter.CTkLabel(frame_2, text="Custom Effects")
    label_3 = customtkinter.CTkLabel(frame_2, text="Contrast")
    label_4 = customtkinter.CTkLabel(frame_2, text="Saturation")
    label_5 = customtkinter.CTkLabel(frame_2, text="Temperature")
    contrast = customtkinter.CTkSlider(frame_2, from_=0, to=100, command=on_contrast_slider_move)
    saturation = customtkinter.CTkSlider(frame_2, from_=0, to=100, command= on_saturation_slider_move)
    color_palette = customtkinter.CTkSlider(frame_2, from_=-100, to=100, command= on_temperature_slider_move)

    label_2.pack()
    label_3.pack()
    contrast.pack()
    label_4.pack()
    saturation.pack()
    label_5.pack()
    color_palette.pack()

#Creates the Filters Window
def filters_ui():
    top = customtkinter.CTkToplevel()
    top.geometry("300x200")
    top.title("Filters")

    frame_1 = customtkinter.CTkFrame(master=top)
    frame_1.pack(pady=10, padx=10, fill="both", expand=True)

    label_1 = customtkinter.CTkLabel(frame_1, text="GrayScale")
    label_2 = customtkinter.CTkLabel(frame_1, text="Noise Reduction")
    label_3 = customtkinter.CTkLabel(frame_1, text="Light Leak")
    label_4 = customtkinter.CTkLabel(frame_1, text="Gamma Correction")
    
    light_light = customtkinter.CTkButton(frame_1, text="Light Leak", command=on_light_leak_slider_move)
    grayscale = customtkinter.CTkSlider(frame_1, from_=0, to=100, command=on_grayscale_slider_move)
    noise = customtkinter.CTkSlider(frame_1, from_=0, to=100, command=on_noise_slider_move)
    gamma = customtkinter.CTkSlider(frame_1, from_=0, to=100, command=on_gamma_slider_move)

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
    top.title("Zoom")

    frame_1 = customtkinter.CTkFrame(master=top)
    frame_1.pack(pady=10, padx=10, fill="both", expand=True)
    label_1 = customtkinter.CTkLabel(frame_1, text="Zoom Level")
    zoom = customtkinter.CTkSlider(frame_1, from_=0, to=100, command=on_zoom_slider_move)

    label_1.pack()
    zoom.pack()

#Creates the Tone Curve Window
def tone_curve_ui():
    top = customtkinter.CTkToplevel()
    top.geometry("300x200")
    top.title("Tone Curve")

    frame_1 = customtkinter.CTkFrame(master=top)
    frame_1.pack(pady=10, padx=10, fill="both", expand=True)
    label_1 = customtkinter.CTkLabel(frame_1, text="Tone Curve")
    tone_curve = customtkinter.CTkSlider(frame_1, from_=0, to=100, command=on_tone_curve_slider_move)

    label_1.pack()
    tone_curve.pack()

#Creates the White Balance Window
def white_balance_ui():
    global original_image_data

    top = customtkinter.CTkToplevel()
    top.geometry("400x200")
    top.title("White Balance")

    frame_1 = customtkinter.CTkFrame(master=top)
    frame_2 = customtkinter.CTkFrame(master=top)
    frame_1.pack(pady=10, padx=10, fill="both", expand=True)
    frame_2.pack(pady=10, padx=10, fill="both", expand=True)
    label_1 = customtkinter.CTkLabel(frame_1, text="Presets")
    label_2 = customtkinter.CTkLabel(frame_2, text="White Balance Level")
    white_button = customtkinter.CTkButton(frame_1, text="White World Assumption", command=white_balance_white)
    grey_button = customtkinter.CTkButton(frame_1, text="Grey World Assumption", command=white_balance_grey)
    balance_slider = customtkinter.CTkSlider(frame_2, from_=0, to=100, command=on_white_balance_slider_move)

    label_1.pack()
    label_2.pack()
    white_button.pack(pady=10, padx=10, side=tk.LEFT)
    grey_button.pack(pady=10, padx=10, side=tk.LEFT)
    balance_slider.pack(pady=10, padx=10)

frame_1 = customtkinter.CTkFrame(master=root)
frame_2 = customtkinter.CTkFrame(master=root)
frame_3 = customtkinter.CTkFrame(master=root)

frame_1.pack(pady=10, padx=10, fill="both", expand=True)
frame_2.pack(pady=10, padx=10, fill="both", expand=True)
frame_3.pack(pady=10, padx=10, fill="both", expand=True)

image_panel = customtkinter.CTkCanvas(master=frame_1, width=width, height=height, bg='white')
image_panel.pack(pady=30, padx=10)
image_panel_image = image_panel.create_image(0, 0, anchor="nw")

# Second Frame Buttons
save_button = customtkinter.CTkButton(frame_2, text="Save Image", command= save_image)
load_button = customtkinter.CTkButton(frame_2, text="Load Image", command=load_image)
quit_button = customtkinter.CTkButton(frame_2, text="Quit", command=root.quit)
reset_button = customtkinter.CTkButton(frame_2, text="Reset Image", command= reset_image)

save_button.pack(side=tk.LEFT, pady=10, padx=20)
load_button.pack(side=tk.LEFT, pady=10, padx=20)
quit_button.pack(side=tk.RIGHT, pady=10, padx=20)
reset_button.pack(side=tk.RIGHT, pady=10, padx=20)

# Third Frame Buttons
film_effects_button = customtkinter.CTkButton(frame_3, text="Film Effects", command=film_effects_ui)
filters_button = customtkinter.CTkButton(frame_3, text="Filters", command=filters_ui)
zoom_button = customtkinter.CTkButton(frame_3, text="Zoom", command=zoom_ui)
white_balance = customtkinter.CTkButton(frame_3, text="White Balance", command=white_balance_ui)
tone_curve = customtkinter.CTkButton(frame_3, text="Tone Curve", command=tone_curve_ui)

film_effects_button.pack(side=tk.LEFT, pady=10, padx=20)
filters_button.pack(side=tk.LEFT, pady=10, padx=20)
zoom_button.pack(side=tk.LEFT, pady=10, padx=20)
white_balance.pack(side=tk.LEFT, pady=10, padx=20)
tone_curve.pack(side=tk.LEFT, pady=10, padx=20)

root.mainloop()