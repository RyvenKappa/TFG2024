import dearpygui.dearpygui as dpg
import time
import multiprocessing as mp

class MyProcess(mp.Process):
    def __init__(self):
        super().__init__()

    def run(self):
        # Realizar alguna tarea
        print("Proceso hijo iniciado")
        while True:
            time.sleep(2)  # Simulando alguna tarea en el proceso hijo
            print("hola")

def main():
    # Inicializar DearPyGUI
    dpg.create_context()
    dpg.create_viewport(title='Dear PyGUI', width=800, height=600)
    dpg.setup_dearpygui()

    # Crear proceso hijo
    p = MyProcess()
    p.start()

    # Mostrar la ventana principal
    dpg.show_viewport()
    dpg.start_dearpygui()
    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()

    # Finalizar DearPyGUI
    dpg.destroy_context()

if __name__ == "__main__":
    main()