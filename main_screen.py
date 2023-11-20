import PySimpleGUI as sg
from PIL import Image
from io import BytesIO
import numpy as np
import cv2


def np_im_to_data(im):
    array = np.array(im, dtype=np.uint8)
    im = Image.fromarray(array)
    with BytesIO() as output:
        im.save(output, format='PNG')
        data = output.getvalue()
    return data

def main_screen():
    width, height = 400, 400

    layout = [
        [
            sg.Graph(
                canvas_size=(width, height),
                graph_bottom_left=(0, 0),
                graph_top_right=(width, height),
                key='-IMAGE-',
                background_color='white',
                change_submits=True,
                drag_submits=True,
                pad=(0, 20),
            ),
        ],

        [
            sg.Button('Save Image'),
            sg.Button('Load Image'),
            sg.Button('Quit')
        ]
    ]

    window = sg.Window("App Title", layout, size=(800, 500), element_justification='c', resizable=True, finalize=True)

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED or event == 'Quit':
            break
        elif event == 'Load Image':
            image_path = sg.popup_get_file('Select an image file', file_types=(("Image files", "*.png;*.jpg;*.jpeg"),))
            if image_path:
                np_image = cv2.imread(image_path)
                np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)

                original_height, original_width, _ = np_image.shape
                target_width, target_height = width, height
                scale_w = target_width / original_width
                scale_h = target_height / original_height
                scale = min(scale_w, scale_h)

                resized_image = cv2.resize(np_image, (int(original_width * scale), int(original_height * scale)))

                np_image_data = np_im_to_data(resized_image)
                window['-IMAGE-'].draw_image(data=np_image_data, location=(0, height))

    window.close()

if __name__ == "__main__":
    main_screen()
