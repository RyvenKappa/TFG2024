"""
Módulo Main para cargar la interfaz e iniciar la aplicación
@author: Diego Aceituno Seoane
@Mail: diego.aceituno@alumnos.upm.es
"""
import dearpygui.dearpygui as dpg
from  WindowManager import Manager
import config_dpg

def main():
    dpg.create_context()
    config_dpg.set_config()#Cargamos la configuración
    manager = Manager()#Creamos el manejador de ventanas
    dpg.create_viewport(title="Auto-NetTest",width=1500,height=750,vsync=True)
    dpg.set_viewport_small_icon("src/images/icono.ico")
    dpg.set_viewport_large_icon("src/images/icono.ico")
    dpg.set_primary_window("MainWindow",True)
    dpg.show_viewport()
    dpg.setup_dearpygui()
    #manager.set_window("DataWindow")
    while dpg.is_dearpygui_running():
        #Actualizar estado de los hilos y de los datos mostrados en pantalla
        manager.update()
        dpg.render_dearpygui_frame()
    dpg.destroy_context()
    manager.general_close_callback()

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()