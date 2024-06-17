import dearpygui.dearpygui as dpg

def set_config():
    #dpg.set_global_font_scale()
    with dpg.font_registry():
        dpg.add_font("src/font/Aptos.ttf",50,tag="LargeFont")
        dpg.add_font("src/font/Aptos.ttf",30,tag="MidFont")
    #Cargado de imagenes dinámicas
    width, height, channels, data = dpg.load_image("src/images/Gamma.png")
    #width1,height1,channels1,data1 = dpg.load_image("src/images/pescado.gif")
    with dpg.texture_registry(show=False):
        dpg.add_static_texture(width=width, height=height, default_value=data, tag="texturaGamma")
        #dpg.add_dynamic_texture(width=width1, height=height1, default_value=data1, tag="pescadoGirando")
    
    #Cargado de registros
    with dpg.value_registry():
        dpg.add_string_value(default_value="",tag="config_user")

