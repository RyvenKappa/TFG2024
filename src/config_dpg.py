import dearpygui.dearpygui as dpg
import numpy as np
def set_config():
    #dpg.set_global_font_scale()
    with dpg.font_registry():
        dpg.add_font("src/font/Aptos.ttf",50,tag="LargeFont")
        dpg.add_font("src/font/Aptos.ttf",30,tag="MidFont")
        dpg.add_font("src/font/Aptos.ttf",20,tag="NormalFont")
        dpg.add_font("src/font/Aptos.ttf",16,tag="SmallFont")
    #Cargado de imagenes dinámicas
    width, height, channels, data = dpg.load_image("src/images/Gamma.png")
    width2, height2, channels2, data2 = dpg.load_image("src/images/LoadingScreen.png")
    width3, height3, channels3, data3 = dpg.load_image("src/images/DataWindowManual.png")
    width4, height4, channels4, data4 = dpg.load_image("src/images/Graficas1DataManual.png")
    width5, height5, channels5, data5 = dpg.load_image("src/images/Graficas2DataManual.png")
    width6, height6, channels6, data6 = dpg.load_image("src/images/EdicionDataManual.png")
    #width1,height1,channels1,data1 = dpg.load_image("src/images/pescado.gif")
    with dpg.texture_registry(show=False):
        dpg.add_static_texture(width=width, height=height, default_value=data, tag="texturaGamma")
        dpg.add_static_texture(width=width2, height=height2, default_value=data2, tag="texturaLoadingManual")
        dpg.add_static_texture(width=width3, height=height3, default_value=data3, tag="DataCompletaManual")
        dpg.add_static_texture(width=width4, height=height4, default_value=data4, tag="Graficas1Manual")
        dpg.add_static_texture(width=width5, height=height5, default_value=data5, tag="Graficas2Manual")
        dpg.add_static_texture(width=width6, height=height6, default_value=data6, tag="EdicionManual")
        dpg.add_raw_texture(width=720,height=480,default_value=np.ones((480*720*3)),tag="VideoTexture",format=dpg.mvFormat_Float_rgb)
        #dpg.add_dynamic_texture(width=width1, height=height1, default_value=data1, tag="pescadoGirando")
    
    #Cargado de registros
    with dpg.value_registry():
        dpg.add_string_value(default_value="",tag="config_user")

    #Configuración de temas para las gráficas
    with dpg.theme(tag="timeline_red_theme"):
        with dpg.theme_component(0):
            dpg.add_theme_color(dpg.mvPlotCol_Fill,(255,0,0,64), category=dpg.mvThemeCat_Plots)
    with dpg.theme(tag="timeline_green_theme"):
        with dpg.theme_component(0):
            dpg.add_theme_color(dpg.mvPlotCol_Fill,(0,255,0,64), category=dpg.mvThemeCat_Plots)
