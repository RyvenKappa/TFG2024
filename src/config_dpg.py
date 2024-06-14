import dearpygui.dearpygui as dpg

def set_config():
    #dpg.set_global_font_scale()
    with dpg.font_registry():
        dpg.add_font("src/font/Aptos.ttf",50,tag="LargeFont")
    #Cargado de imagenes
    width, height, channels, data = dpg.load_image("src/Gamma.png")
    with dpg.texture_registry(show=False):
        dpg.add_static_texture(width=width, height=height, default_value=data, tag="texturaGamma")
    #Cargado de registros
    with dpg.value_registry():
        dpg.add_string_value(default_value="",tag="config_user")

