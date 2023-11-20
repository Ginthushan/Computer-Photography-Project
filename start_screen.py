import PySimpleGUI as sg
from main_screen import main_screen

def start_screen():
    layout = [
        [sg.Text("Welcome to 'Application Title'", font=("Comic Sans", 16), justification='center')],
        [sg.Button("Start", size=(20, 2), key="-START-")],
        [sg.Button("About", size=(20, 2), key="-ABOUT-")]
    ]

    window = sg.Window("Home Screen", layout, size=(400, 200), element_justification='c')
    

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED:
            break
        elif event == "-START-":
            window.close()
            main_screen()
        elif event == "-ABOUT-":
            sg.popup("Photoshop Clone\nVersion 1.0\n\nDeveloped by Ginthushan Kandasamy & Dennis Peng")


    window.close()