"""
Módulo Main para cargar la interfaz e iniciar la aplicación
@author: Diego Aceituno Seoane
@Mail: diego.aceituno@alumnos.upm.es
"""
import dearpygui.dearpygui as dpg

dpg.create_context()
dpg.create_viewport(title="Auto-NetTest",width=1000,height=800,vsync=True)