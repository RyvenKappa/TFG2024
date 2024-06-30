"""
Modulo que contiene las diferentes ventanas de la aplicación, incluyendo las transiciones entre las mismas
@author: Diego Aceituno Seoane
@Mail: diego.aceituno@alumnos.upm.es
"""
import dearpygui.dearpygui as dpg
from yolo import model
import multiprocessing as mp
from results_module import Data_Processor
import cv2 as cv
import pandas as pd
import numpy as np
import json
import csv
from video_controller import Controller

class Manager():
    """
    Clase que implementa la funcionalidad de las diferentes ventanas de la aplicación
    """
    def __init__(self) -> None:
        self.active_window = None
        self.data_mode = False
        self.window_tags=[]
        self.infiriendo = False
        self.raw_data = None
        self.dataset_global_left = None
        self.dataset_global_right = None
        self.save_path = ""
        self.total_frames = 0
        self.first_image = None
        self.actual_image = None
        self.prob_mov_left = []
        self.prob_mov_right = []
        self.real_mov_left = []
        self.real_mov_right = []
        self.clicked_left = None
        self.clicked_right = None
        self.new_click = False
        self.video_name = None
        self.left_frames = []
        self.right_frames = []
        self.video_controller = None
        self.clicked_stem_left = None
        self.clicked_stem_right = None
        self.eje_frame = np.linspace(start=1,stop=190,num=190) # Eje simple para el stado inicial de las gráficas 
        #Handler para los clicks sobre diversos elementos
        with dpg.item_handler_registry(tag="Handlers") as handlers:
            dpg.add_item_clicked_handler(callback=self.clicked_callback)
        self.start_screen()

    def start_screen(self) -> None:
        with dpg.window(tag="MainWindow",no_scrollbar=True,autosize=True,show=False):
            self.window_tags.append("MainWindow")
            self.set_window("MainWindow")
            with dpg.child_window(autosize_x=True,height=150):
                with dpg.table(tag="UserTable",resizable=False,header_row=False,reorderable=True):
                    dpg.add_table_row()
                    dpg.add_table_column(width_stretch=True)
                    dpg.add_table_column(tag="Columna2")
                    with dpg.table_row():
                        with dpg.child_window(autosize_x=True,height=120,border=False):
                            boton = dpg.add_button(label="Manual de usuario",callback=self.show_manual)
                            dpg.bind_item_font(boton,"MidFont")
                            with dpg.window(label="Error al intentar inferir",modal=True,show=False,tag="ModalWindowError"):
                                texto = dpg.add_text("",tag="TextoError")
                                dpg.bind_item_font(texto,"MidFont")
                            with dpg.window(label="Manual de usuario", modal=True, show=False, tag="ManualModalWindow",width=1000,height=400):
                                titulo = dpg.add_text("Aplicación de automatización del experimento NetTest")
                                dpg.bind_item_font(titulo,"NormalFont")
                                dpg.add_text("Parte del TFG de Diego Aceituno Seoane",bullet=True)
                                dpg.add_text("diego.aceituno@alumnos.upm.es",bullet=True)
                                with dpg.collapsing_header(label="Ventana de inicio y limitaciones sobre el video"):
                                    dpg.add_text("Esta ventana permite seleccionar el fichero que vamos a utilizar sobre el cual vamos a realizar la inferencia.\n"+
                                                 "La selección de este fichero esta limitada por el tipo de archivo soportado. Las extensiones de archivos soportadas son:")
                                    dpg.add_text(".asf",bullet=True)
                                    dpg.add_text(".avi",bullet=True)
                                    dpg.add_text(".gif",bullet=True)
                                    dpg.add_text(".m4v",bullet=True)
                                    dpg.add_text(".mkv",bullet=True)
                                    dpg.add_text(".mov",bullet=True)
                                    dpg.add_text(".mp4",bullet=True)
                                    dpg.add_text(".mpeg",bullet=True)
                                    dpg.add_text(".mpg",bullet=True)
                                    dpg.add_text(".ts",bullet=True)
                                    dpg.add_text(".wmv",bullet=True)
                                    dpg.add_text(".webm",bullet=True)
                                    dpg.add_text("Además de esta limitación sobre el formato, se exige al usuario que el video seleccionado cumpla 2 condiciones:")
                                    dpg.add_text("El video debe estar en local, OneDrive no permite un funcionamiento correcto",bullet=True)
                                    dpg.add_text("El video no debe durar menos de 50 fotogramas, esto se utiliza para estimar si exiten 1 o 2 truchas.",bullet=True)
                                    dpg.add_text("Aparte de esto, es recomendable que el video sea 16:9 en relación de aspecto y el pez no ocupe toda la pantalla.")
                                    dpg.add_spacer()
                                    text = dpg.add_text("Una vez se seleccione el fichero, aparecera el botón para pasar a realizar la inferencia.")
                                    dpg.bind_item_font(text,"NormalFont")
                                with dpg.collapsing_header(label="Inferencia y pantalla de carga"):
                                    text = dpg.add_text("Esta pantalla muestra el progreso de la inferencia sobre el video administrado.\n"+
                                                 "Aparte de esto, nos permite volver a la pantalla anterior si hemos seleccionado un video erroneo o ya parametrizado.\n"+
                                                 "En cuanto el progreso finalize, se pasará directamente a la ventana de análisis de datos")
                                    dpg.bind_item_font(text,"NormalFont")
                                    dpg.add_image(texture_tag="texturaLoadingManual")
                                with dpg.collapsing_header(label="Pantalla de datos"):
                                    texto = dpg.add_text("Esta pantalla muestra los resultados obtenidos de la inferencia del video. Además permite el editado y guardado de los mismos.")
                                    dpg.bind_item_font(texto,"NormalFont")
                                    dpg.add_text("En esta pantalla hay muchos elementos que se pueden mover horizontalmente para permitir una mejor calidad de visión de los resultados.")
                                    dpg.add_image(texture_tag="DataCompletaManual")
                                    dpg.add_spacer(height=30)
                                    texto2 = dpg.add_text("En la primera sección podemos observar diferentes parámetros para los dos peces organizados por pestañas:")
                                    dpg.bind_item_font(texto2,"NormalFont")
                                    dpg.add_text("Porcentaje del area de la imagen ocupada por la trucha.",bullet=True)
                                    dpg.add_text("Movimiento absoluto del centroide de la trucha detectada.",bullet=True)
                                    dpg.add_text("Relación ancho alto de la trucha detectada.",bullet=True)
                                    dpg.add_text("Varianza del blur en la zona en la que se ha detectado la trucha. Cuanto menor, mas borrosa es la imagen.",bullet=True)
                                    dpg.add_image(texture_tag="Graficas1Manual")
                                    dpg.add_spacer(height=30)
                                    texto2 = dpg.add_text("En la segunda sección podemos observar los anteriores parámetros unidos en una sola gráfica, para poder ver patrones.\n"+
                                                          "Aparte de esto, podemos ver el número de movimientos estimados que ha realizado cada pez")
                                    dpg.bind_item_font(texto2,"NormalFont")
                                    dpg.add_text("En las gráficas se puede usar la leyenda para activar y desactivar diferentes datos.")
                                    dpg.add_image(texture_tag="Graficas2Manual")
                                    dpg.add_spacer(height=30)
                                    texto2 = dpg.add_text("En la tercera sección podemos editar y verificar los resultados a través de usar las diferentes timelines para cada trucha.\n"+
                                                          "A través del marcador que coloquemos, podemos seleccionar añadir o eliminar movimiento(si ha habido alguno en ese punto).")
                                    dpg.bind_item_font(texto2,"NormalFont")
                                    dpg.add_image(texture_tag="EdicionManual")
                        dpg.add_image(texture_tag="texturaGamma",tag="ImagenGamma")
            with dpg.table(tag="MainTable",resizable=False,header_row=False,reorderable=True):
                dpg.add_table_row()
                dpg.add_table_row()
                dpg.add_table_row()
                dpg.add_table_row()
                dpg.add_table_row()
                dpg.add_table_column(width_stretch=True)
                with dpg.table_row():
                    textoSeleccion = dpg.add_text(default_value="Seleccione el video sobre el que realizar la inferencia:")
                    dpg.bind_item_font(textoSeleccion,"NormalFont")
                with dpg.table_row():
                    botonSeleccionar = dpg.add_button(label="Seleccionar fichero", callback=lambda:dpg.configure_item("__Explorador",show=True))
                    dpg.bind_item_font(botonSeleccionar,"NormalFont")
                with dpg.table_row():
                    i_text = dpg.add_input_text(tag="inputText1",width=400,readonly=True)
                    dpg.bind_item_font(i_text,"NormalFont")
                    with dpg.file_dialog(
                            label="Explorador",
                            directory_selector=False,
                            show=False,
                            modal=False,
                            tag="__Explorador",
                            callback=self.file_selected_callback,
                            cancel_callback=self.no_file_selected_callback,
                            height=300,
                            width=400
                        ):
                        dpg.add_file_extension("Archivos de video{.asf,.avi,.gif,.m4v,.mkv,.mov,.mp4,.mpeg,.mpg,.ts,.wmv,.webm}")#Videos
                        dpg.add_file_extension('.*')
                with dpg.table_row():
                    dpg.add_spacer()
                with dpg.table_row():
                    with dpg.table(resizable=False,header_row=False,reorderable=True):
                        dpg.add_table_row()
                        dpg.add_table_column()
                        dpg.add_table_column(width_stretch=True)
                        dpg.add_table_column()
                        with dpg.table_row():
                            dpg.add_spacer()
                            b_inferir = dpg.add_button(width=-1,height=150,label="Inferir",show=False,tag="BotonInferir",callback=self.inferir_callback)
                            dpg.bind_item_font(b_inferir,"LargeFont")
                            dpg.add_spacer()
        with dpg.window(tag="LoadingWindow",no_scrollbar=True,autosize=True,show=False):
            self.window_tags.append("LoadingWindow")
            with dpg.child_window(autosize_x=True,height=200):
                with dpg.table(tag="LoadingTable",resizable=False,header_row=False,reorderable=True):
                    dpg.add_table_row()
                    dpg.add_table_row()
                    dpg.add_table_row()
                    dpg.add_table_column()
                    dpg.add_table_column()
                    dpg.add_table_column()
                    with dpg.table_row():
                                dpg.add_spacer()
                                with dpg.table(resizable=False,header_row=False,reorderable=True):
                                    dpg.add_table_row()
                                    dpg.add_table_column()
                                    dpg.add_table_column()
                                    dpg.add_table_column()
                                    with dpg.table_row():
                                        dpg.add_spacer(tag="LeftSpacerInfiriendo")
                                        dpg.add_loading_indicator(radius=7,tag="LoadingIcon")
                                        dpg.add_spacer(tag="RightSpacerInfiriendo")
                                dpg.add_spacer()
                    with dpg.table_row():
                        dpg.add_spacer()
                        dpg.add_progress_bar(label="Infiriendo...", tag="progreso",default_value=0.0,width=-1)
                        dpg.add_spacer()
                    with dpg.table_row():
                        dpg.add_spacer()
                        texto_carga = dpg.add_text(default_value=f"0 fotogramas de 0 procesados",label="Texto de progreso",tag="TextProgreso")
                        dpg.bind_item_font(texto_carga,"MidFont")
                        dpg.add_spacer()
            boton_cancelar = dpg.add_button(width=-1,height=150,label="Cancelar Inferencia",show=True,tag="BotonCancelarInferencia",callback=self.cancelar_callback)
            dpg.bind_item_font(boton_cancelar,"MidFont")
        with dpg.window(tag="DataWindow",no_scrollbar=True,autosize=True,show=False):
            self.window_tags.append("DataWindow")
            with dpg.table(tag="DataTable",resizable=True,header_row=False,reorderable=True):
                dpg.add_table_row()
                dpg.add_table_column()
                dpg.add_table_column()
                with dpg.table_row():
                    with dpg.child_window(no_scrollbar=True,height=-1,border=False):
                        boton_cancelar = dpg.add_button(width=-1,height=30,label="Volver a la pantalla inicial",show=True,tag="BotonVolverAtras",callback=self.volver_callback)
                        #dpg.bind_item_font(boton_cancelar,"MidFont")
                        with dpg.table(tag="GraphTable",resizable=True,header_row=False,reorderable=True):
                            dpg.add_table_row()
                            dpg.add_table_row()
                            dpg.add_table_column()
                            with dpg.table_row():
                                with dpg.child_window(no_scrollbar=True,height=500,tag="ChildWindowGraphs"):
                                    with dpg.tab_bar():
                                        with dpg.tab(label="Datos de Area"):

                                            with dpg.plot(label="Cambio en las areas",width=-1,height=-1,anti_aliased=True,tag="AreasPlot"):
                                                dpg.add_plot_legend()
                                                #Las dos comparten eje x
                                                dpg.add_plot_axis(dpg.mvXAxis,label="Frame",tag="area_x")
                                                
                                                with dpg.plot_axis(dpg.mvYAxis,label="area_izquierda",tag="AreaYIzquierda"):
                                                    dpg.add_line_series(self.eje_frame,y=np.zeros(190),label="area_izquierda",tag="Area_Izquierda")

                                                with dpg.plot_axis(dpg.mvYAxis,label="area_derecha",tag="AreaYDerecha"):
                                                    dpg.add_line_series(self.eje_frame,y=np.ones(190),label="area_derecha",tag="Area_Derecha")
                                        
                                        with dpg.tab(label="Datos del centroide"):

                                            with dpg.plot(label="Cambio en los centroides",width=-1,height=-1,anti_aliased=True,tag="CentroidesPlot"):
                                                dpg.add_plot_legend()
                                                #Las dos comparten eje x
                                                dpg.add_plot_axis(dpg.mvXAxis,label="Frame",tag="centroide_x")
                                                
                                                with dpg.plot_axis(dpg.mvYAxis,label="centroide_izquierda",tag="CentroideYIzquierda"):
                                                    dpg.add_line_series(self.eje_frame,y=np.zeros(190),label="centroide_izquierda",tag="Centroide_Izquierda")

                                                with dpg.plot_axis(dpg.mvYAxis,label="area_derecha",tag="CentroideYDerecha"):
                                                    dpg.add_line_series(self.eje_frame,y=np.ones(190),label="centroide_derecha",tag="Centroide_Derecha")
                                        
                                        with dpg.tab(label="Cambio de la relación altura ancho"):

                                            with dpg.plot(label="Cambio en la relación ancho-altura",width=-1,height=-1,anti_aliased=True,tag="RelacionPlot"):
                                                dpg.add_plot_legend()
                                                #Las dos comparten eje x
                                                dpg.add_plot_axis(dpg.mvXAxis,label="Frame",tag="hw_x")
                                                
                                                with dpg.plot_axis(dpg.mvYAxis,label="relación_izquierda",tag="RelacionYI"):
                                                    dpg.add_line_series(self.eje_frame,y=np.zeros(190),label="relación_izquierda",tag="Relacion_Izquierda")

                                                with dpg.plot_axis(dpg.mvYAxis,label="area_derecha",tag="RelacionYD"):
                                                    dpg.add_line_series(self.eje_frame,y=np.ones(190),label="relación_derecha",tag="Relacion_Derecha")
                                        
                                        with dpg.tab(label="Cambio del blur pez izquierda"):

                                            with dpg.plot(label="Cambio en el blur",width=-1,height=-1,anti_aliased=True,tag="BlurPlot"):
                                                dpg.add_plot_legend()
                                                #Las dos comparten eje x
                                                dpg.add_plot_axis(dpg.mvXAxis,label="Frame",tag="blur_x")
                                                
                                                with dpg.plot_axis(dpg.mvYAxis,label="blur_izquierda",tag="BlurYI"):
                                                    dpg.add_line_series(self.eje_frame,y=np.zeros(190),label="blur_izquierda",tag="Blur_Izquierda")

                                                with dpg.plot_axis(dpg.mvYAxis,label="blur_derecha",tag="BlurYD"):
                                                    dpg.add_line_series(self.eje_frame,y=np.ones(190),label="blur_derecha",tag="Blur_Derecha")
                                        
                            with dpg.table_row():
                                with dpg.table(tag="GraphTableFinal",resizable=True,header_row=False,reorderable=True):
                                    dpg.add_table_row()
                                    dpg.add_table_row()
                                    dpg.add_table_column()
                                    dpg.add_table_column()
                                    with dpg.table_row():
                                        with dpg.plot(label="Datos completos pez izquierda",width=-1,height=-1,tag="GraficaFinalIzquierda",anti_aliased=True):
                                                dpg.add_plot_legend()
                                                #Las dos comparten eje x
                                                dpg.add_plot_axis(dpg.mvXAxis,label="Frame",tag="xCompletaIzquierda")
                                                
                                                with dpg.plot_axis(dpg.mvYAxis,label="area_izquierda",tag="YAreaI"):
                                                    dpg.add_line_series(self.eje_frame,y=np.zeros(190),label="area_izquierda",tag="Area_IzquierdaCompleta")

                                                with dpg.plot_axis(dpg.mvYAxis,label="centroide_izquierda",tag="YCentroideI"):
                                                    dpg.add_line_series(self.eje_frame,y=np.zeros(190),label="centroide_izquierda",tag="Centroide_IzquierdaCompleta")

                                                with dpg.plot_axis(dpg.mvYAxis,label="blur_izquierda",tag="YBlurI"):
                                                    dpg.add_line_series(self.eje_frame,y=np.zeros(190),label="blur_izquierda",tag="Blur_IzquierdaCompleta")

                                        with dpg.plot(label="Datos completos pez derecha",width=-1,height=-1,tag="GraficaFinalDerecha",anti_aliased=True):
                                                dpg.add_plot_legend()
                                                #Las dos comparten eje x
                                                dpg.add_plot_axis(dpg.mvXAxis,label="Frame",tag="xCompletaDerecha")
                                                
                                                with dpg.plot_axis(dpg.mvYAxis,label="area_derecha",tag="YAreaD"):
                                                    dpg.add_line_series(self.eje_frame,y=np.zeros(190),label="area_derecha",tag="Area_DerechaCompleta")

                                                with dpg.plot_axis(dpg.mvYAxis,label="centroide_derecha",tag="YCentroideD"):
                                                    dpg.add_line_series(self.eje_frame,y=np.zeros(190),label="centroide_derecha",tag="Centroide_DerechaCompleta")

                                                with dpg.plot_axis(dpg.mvYAxis,label="blur_derecha",tag="YBlurD"):
                                                    dpg.add_line_series(self.eje_frame,y=np.zeros(190),label="blur_derecha",tag="Blur_DerechaCompleta")

                                    with dpg.table_row():
                                        text = dpg.add_text("Numero total de movimientos del pez izquierdo: ",tag="MovimientosIzquierda",color=(255,0,0,255))
                                        dpg.bind_item_font(text,"NormalFont")
                                        text = dpg.add_text("Numero total de movimientos del pez derecho: ",tag="MovimientosDerecha",color=(255,0,0,255))
                                        dpg.bind_item_font(text,"NormalFont")
                    with dpg.child_window(no_scrollbar=True,height=-1,border=False):
                        with dpg.child_window(no_scrollbar=True,height=-300,tag="EditarWindow"):
                            with dpg.child_window(no_scrollbar=True,width=-1,height=100,tag="VideoWindow"):
                                dpg.add_image("VideoTexture",tag="Imagen")
                            with dpg.group():
                                with dpg.table(resizable=False,header_row=False,reorderable=True,tag="TablaControl"):
                                    dpg.add_table_row(tag="PlayButtonRow")
                                    dpg.add_table_row(tag="TimelineRow")
                                    dpg.add_table_row(tag="BotonesAddDeleteRow")
                                    dpg.add_table_column()
                                    with dpg.table_row():
                                        with dpg.table(resizable=False,header_row=False,reorderable=True):
                                            dpg.add_table_column()
                                            dpg.add_table_column()
                                            dpg.add_table_column()
                                            with dpg.table_row():
                                                dpg.add_spacer(width=50,tag="LeftSpacerPlay")
                                                dpg.add_button(label="PLAY",width=-1,callback=self.play_button_callback,tag="PlayStopButton")
                                                dpg.add_spacer(width=50,tag="RightSpacerPlay")
                                    with dpg.table_row():
                                        with dpg.table(resizable=True,header_row=False,reorderable=True):
                                            dpg.add_table_column()
                                            dpg.add_table_column()
                                            with dpg.table_row():
                                                with dpg.plot(width=-1, height=-1,tag="TimeLinePlot"):

                                                    # Añadir ejes
                                                    x_axis = dpg.add_plot_axis(dpg.mvXAxis,tag="TimeLineXIzquierda")
                                                    with dpg.plot_axis(dpg.mvYAxis,no_tick_labels=True,no_gridlines=True,no_tick_marks=True,tag="y_axis"):
                                                        # Añadir la serie de sombreado
                                                        dpg.add_shade_series(self.eje_frame,y1=np.ones(190),tag="ZonasConFrame",parent="y_axis")  # Rojo
                                                        dpg.bind_item_theme(dpg.last_item(),"timeline_red_theme")
                                                        dpg.set_axis_limits("y_axis",0,1)
                                                        dpg.add_shade_series(self.eje_frame,y1=np.ones(190),tag="ZonasSinFrame",parent="y_axis")  # Verde
                                                        dpg.bind_item_theme(dpg.last_item(),"timeline_green_theme")
                                                        dpg.set_axis_limits("y_axis",0,1)
                                                        dpg.add_stem_series(self.eje_frame,np.ones(190),tag="ClickedStem",parent="y_axis")
                                                dpg.bind_item_handler_registry("TimeLinePlot","Handlers")

                                                with dpg.plot(width=-1, height=-1,tag="TimeLinePlot2"):

                                                    # Añadir ejes
                                                    x_axis = dpg.add_plot_axis(dpg.mvXAxis,tag="TimeLineXDerecha")
                                                    with dpg.plot_axis(dpg.mvYAxis,no_tick_labels=True,no_gridlines=True,no_tick_marks=True,tag="y_axis2"):
                                                        # Añadir la serie de sombreado
                                                        dpg.add_shade_series(self.eje_frame,y1=np.ones(190),tag="ZonasConFrame2",parent="y_axis2")  # Rojo
                                                        dpg.bind_item_theme(dpg.last_item(),"timeline_red_theme")
                                                        dpg.set_axis_limits("y_axis2",0,1)
                                                        dpg.add_shade_series(self.eje_frame,y1=np.ones(190),tag="ZonasSinFrame2",parent="y_axis2")  # Verde
                                                        dpg.bind_item_theme(dpg.last_item(),"timeline_green_theme")
                                                        dpg.set_axis_limits("y_axis2",0,1)
                                                        dpg.add_stem_series(self.eje_frame,np.ones(190),tag="ClickedStem2",parent="y_axis2")

                                                dpg.bind_item_handler_registry("TimeLinePlot2","Handlers")

                                    with dpg.table_row():
                                        with dpg.table(resizable=False,header_row=False,reorderable=True):
                                            dpg.add_table_row()
                                            dpg.add_table_column()
                                            dpg.add_table_column()
                                            dpg.add_table_column()
                                            dpg.add_table_column()
                                            with dpg.table_row():
                                                dpg.add_button(width=-1,label="Añadir a la Izquierda",tag="AddLeft",callback=self.add_left_callback)
                                                dpg.add_button(width=-1,label="Eliminar en la Izquierda",tag="DelLeft",callback=self.del_left_callback)
                                                dpg.add_button(width=-1,label="Añadir a la Derecha",tag="AddRight",callback=self.add_right_callback)
                                                dpg.add_button(width=-1,label="Eliminar en la Derecha",tag="DelRight",callback=self.del_right_callback)
                        with dpg.table(resizable=False,header_row=False,reorderable=True):
                            dpg.add_table_row()
                            dpg.add_table_column()
                            dpg.add_table_column()
                            with dpg.table_row():
                                with dpg.group():
                                    dpg.add_button(label="Seleccionar carpeta", callback=lambda:dpg.configure_item("__Explorador2",show=True))
                                    with dpg.file_dialog(
                                        label="Explorador",
                                        directory_selector=True,
                                        show=False,
                                        modal=False,
                                        tag="__Explorador2",
                                        callback=self.folder_selected_callback,
                                        cancel_callback=self.no_folder_selected_callback,
                                        height=500,
                                        width=500
                                    ):
                                        dpg.add_file_extension("", color=(255,255,255,255))
                                dpg.add_button(label="Guardar Resultados",width=-1,height=-1,callback=self.save_results_callback)

    def show_manual(self,sender,app_data):
        """
            Método callback para enseñar el manual de usuario en un PopUp
        """
        dpg.show_item("ManualModalWindow")
        pass

    def file_selected_callback(self,sender,app_data):
        """
            Callback para archivo seleccionado, registra el archivo seleccionado y lo pone en el inputText1
            Además, hace visible el botón para realizar la inferencia.
        """
        dpg.configure_item("inputText1",default_value=(f"{next(iter(app_data['selections'].values()))}"))
        dpg.configure_item("BotonInferir",show=True)

    def no_file_selected_callback(self,sender,app_data):
        """
            Callback para no archivo seleccionado
        """
        dpg.configure_item("inputText1",default_value="")
        dpg.configure_item("BotonInferir",show=False)
    
    def folder_selected_callback(self,sender,app_data):
        """
            Método que con una carpeta seleccionada, la añade como fichero base
        """
        self.save_path = app_data.get('current_path')+"\\"

    def no_folder_selected_callback(self,sender,app_data):
        """
            Callback si se cancela la selección de carpeta, por defecto guardamos en la raiz donde esta el exe
        """
        self.save_path = ""

    def inferir_callback(self,sender,app_data):
        """
            Callback para el botón de inferencia, verifica el video, inicia los procesos, inicia la inferencia y cambia de pantalla
        """
        self.video_path:str = dpg.get_value('inputText1')
        self.cap = cv.VideoCapture(self.video_path)
        self.total_frames = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        if self.total_frames <51:
            dpg.show_item("ModalWindowError")
            dpg.set_value("TextoError","Error, el video dura menos de 50 fotogramas")
            self.cap.release()
            return
        ret, self.first_image = self.cap.read()#Nos guardamos la primera imagen para futuro
        if not ret:
            self.first_image = None
            self.cap.release()

        results_pipe = mp.Pipe(duplex=False) #(Output,Input) de una conexión unidireccional entre el proceso main y el proceso de resultados
        inference_pipe = mp.Pipe(duplex=False) #Tubería para que el proceso de inferencia le vaya pasando las imágenes al de procesado de resultados

        self.model = model()
        self.data_processor = Data_Processor(inference_pipe[0],results_pipe[1],self.total_frames)
        self.frame_enpoint = results_pipe[0]
        self.model.video_inference(inference_pipe[1],self.video_path)
        self.data_processor.start()
        self.infiriendo = True
        dpg.set_value(item="TextProgreso",value=f"0 fotogramas de {self.total_frames} procesados")
        dpg.set_value(item="progreso",value=0.0)

        self.video_name = self.video_path[self.video_path.rfind("\\")+1:]

        #Cambio de pantalla
        self.set_window("LoadingWindow")

    def cancelar_callback(self,sender,app_data):
        """
            Callback para cancelar la inferencia
        """
        try:
            self.data_processor.terminate()
            self.data_processor.close()
        except:
            pass
        finally:
            self.model.stop_video_inference()
            self.set_window("MainWindow")
            self.infiriendo = False

    def volver_callback(self,sender,app_data):
        """
            Método callback para cerrar el proceso de obtención de imagenes y volver a la pantalla inicial
        """
        try:
            self.video_controller.terminate()
            self.video_controller.close()
        except:
            pass
        finally:
            dpg.configure_item("PlayStopButton",label="PLAY")
            self.set_window("MainWindow")


    def save_results_callback(self,sender,app_data):
        """
            Método que guarda los resultados de la inferencia en un formato común, de manera que se pueda utilizar más tarde
        """
        path = self.save_path+self.video_name + ".csv"
        try:
            archivo_csv = open(path, mode='w', newline='')
            writer = csv.writer(archivo_csv, delimiter=',')
            writer.writerow([self.video_name])
            writer.writerow([len(self.left_frames)])
            writer.writerow(self.left_frames)
            if type(self.dataset_global_right)!= type(None):
                writer.writerow([len(self.right_frames)])
                writer.writerow(self.right_frames)
            archivo_csv.close()
            diccionario = {
                "video_name": self.video_name,
                "left":{
                    "movement_number":len(self.left_frames),
                    "movement_frames":self.left_frames
                },
                "right": None
            }
            if type(self.dataset_global_right)!= type(None):
                diccionario["right"]={
                                    "movement_number":len(self.right_frames),
                                    "movement_frames":self.right_frames
                                    }
            path = self.save_path+self.video_name + ".json"
            archivo = open(path,mode="w")
            json.dump(diccionario,archivo,indent=4)
        except Exception as e:
            print(e)
    
    def play_button_callback(self,sender,app_data):
        """
            Método para controlar el pulsado del play sobre el video
        """
        label = dpg.get_item_configuration(sender)['label']
        if label == "PLAY":
            self.control_pipe_endpoint.send(True)
            dpg.configure_item(sender,label="STOP")
        else:
            self.control_pipe_endpoint.send(False)
            dpg.configure_item(sender,label="PLAY")

    def add_left_callback(self,sender,app_data):
        """
            Método callback para añadir manualmente un movimiento y región cercana en la izquierda
        """
        #A través del valor interno marcamos esa zona y actualizamos las gráficas
        if self.clicked_left is not None:
            if self.clicked_left not in self.left_frames:
                zone = [self.clicked_left-2,self.clicked_left-1,self.clicked_left,self.clicked_left+1,self.clicked_left+2]
                self.left_moves = self.left_moves + 1
                self.left_frames.append(self.clicked_left)
                self.left_frames.sort()
                for i in zone:
                    self.real_mov_left[i] = 1
                    self.prob_mov_left[i] = 0
                dpg.set_value("MovimientosIzquierda",f"Numero total de movimientos del pez derecho:\n{self.left_moves} movimientos")
                dpg.set_value("ZonasConFrame",[self.eje_frame,self.real_mov_left,np.zeros(self.real_mov_left.shape)])
                dpg.set_value("ZonasSinFrame",[self.eje_frame,self.prob_mov_left,np.zeros(self.prob_mov_left.shape)])

    def add_right_callback(self,sender,app_data):
        """
            Método callback para añadir manualmente un movimiento y región cercana en la derecha
        """
        if self.clicked_right is not None:
            if self.clicked_right not in self.right_frames:
                zone = [self.clicked_right-2,self.clicked_right-1,self.clicked_right,self.clicked_right+1,self.clicked_right+2]
                self.right_moves = self.right_moves + 1
                self.right_frames.append(self.clicked_right)
                self.right_frames.sort()
                for i in zone:
                    self.real_mov_right[i] = 1
                    self.prob_mov_right[i] = 0
                dpg.set_value("MovimientosDerecha",f"Numero total de movimientos del pez derecho:\n{self.right_moves} movimientos")
                dpg.set_value("ZonasConFrame2",[self.eje_frame,self.real_mov_right,np.zeros(self.real_mov_right.shape)])
                dpg.set_value("ZonasSinFrame2",[self.eje_frame,self.prob_mov_right,np.zeros(self.prob_mov_right.shape)])
    
    def del_left_callback(self,sender,app_data):
        """
            Método callback para eliminar manualmente un movimiento y su región en la izquierda
        """
        if self.clicked_left is not None:
            zone = [self.clicked_left-2,self.clicked_left-1,self.clicked_left,self.clicked_left+1,self.clicked_left+2]
            for i in zone:
                if i in self.left_frames:
                    zone2 = list(range(i-5,i+5))
                    self.left_moves = self.left_moves - 1
                    self.left_frames.remove(i)
                    for i2 in zone2:
                            self.real_mov_left[i2] = 0
            dpg.set_value("MovimientosIzquierda",f"Numero total de movimientos del pez izquierdo:\n{self.left_moves} movimientos")
            dpg.set_value("ZonasConFrame",[self.eje_frame,self.real_mov_left,np.zeros(self.real_mov_left.shape)])
            dpg.set_value("ZonasSinFrame",[self.eje_frame,self.prob_mov_left,np.zeros(self.prob_mov_left.shape)])

    def del_right_callback(self,sender,app_data):
        """
            Método callback para eliminar manualmente un movimiento y su región en la derecha
        """
        if self.clicked_right is not None:
            zone = [self.clicked_right-2,self.clicked_right-1,self.clicked_right,self.clicked_right+1,self.clicked_right+2]
            for i in zone:
                if i in self.right_frames:
                    #zone2 = [i-2,i-1,i,i+1,i+2]
                    zone2 = list(range(i-5,i+5))
                    self.right_moves = self.right_moves - 1
                    self.right_frames.remove(i)
                    for i2 in zone2:
                            self.real_mov_right[i2] = 0
            dpg.set_value("MovimientosDerecha",f"Numero total de movimientos del pez derecho:\n{self.right_moves} movimientos")
            dpg.set_value("ZonasConFrame2",[self.eje_frame,self.real_mov_right,np.zeros(self.real_mov_right.shape)])
            dpg.set_value("ZonasSinFrame2",[self.eje_frame,self.prob_mov_right,np.zeros(self.prob_mov_right.shape)])

    def clicked_callback(self,sender,app_data):
        """
            Método para registrar el frame seleccionado
        """
        self.clicked_stem_left = np.full(self.total_frames,-1)
        self.clicked_stem_right = np.full(self.total_frames,-1)
        self.clicked_left = None
        self.clicked_right = None
        self.frame_clicked = 0
        item_clickeado = dpg.get_item_alias(app_data[1])
        if item_clickeado == "TimeLinePlot":
            self.clicked_left = round(dpg.get_plot_mouse_pos()[0])
            self.frame_clicked = self.clicked_left
            self.new_click = True
            self.clicked_stem_left[round(dpg.get_plot_mouse_pos()[0])] = 2
        elif item_clickeado == "TimeLinePlot2":
            self.clicked_right = round(dpg.get_plot_mouse_pos()[0])
            self.frame_clicked = self.clicked_right
            self.new_click = True
            self.clicked_stem_right[round(dpg.get_plot_mouse_pos()[0])] = 2
        dpg.set_value("ClickedStem",[self.eje_frame,self.clicked_stem_left])
        dpg.set_value("ClickedStem2",[self.eje_frame,self.clicked_stem_right])
        #Paramos el video, cambiamos la label y además cambiamos la posición del video para que vaya a ese momento
        self.control_pipe_endpoint.send(self.frame_clicked)
        dpg.configure_item("PlayStopButton",label="PLAY")



    def update(self):
        """
            Método que sirve para la actualización de las variables temporales y la comunicación con los procesos de inferencia.
            Además sirve para la actualización de texturas
        """
        #Ajustamos el tamaño de algunos elementos para que se ajusten al tamaño de la ventana
        if self.data_mode:
            dpg.set_item_height("ChildWindowGraphs",dpg.get_item_rect_size(self.active_window)[1]/2)
            #Gráficas totales izquierda y derecha
            dpg.set_item_height("GraficaFinalIzquierda",dpg.get_item_rect_size(self.active_window)[1]/3)
            dpg.set_item_height("GraficaFinalDerecha",dpg.get_item_rect_size(self.active_window)[1]/3)
            #ChildWindow para editar
            dpg.set_item_height("EditarWindow",dpg.get_item_rect_size(self.active_window)[1]*3.5/4)
            #ChildWindow en el que va la imgen pura
            dpg.set_item_height("VideoWindow",dpg.get_item_rect_size("EditarWindow")[1]*3/4)
            #Botón play
            dpg.set_item_width("LeftSpacerPlay",dpg.get_item_rect_size("EditarWindow")[0]/4)
            dpg.set_item_width("RightSpacerPlay",dpg.get_item_rect_size("EditarWindow")[0]/4)
            #TimeLine
            dpg.set_item_height("TimeLinePlot",dpg.get_item_rect_size("EditarWindow")[1]/7.5)
            dpg.set_item_height("TimeLinePlot2",dpg.get_item_rect_size("EditarWindow")[1]/7.5)
            if self.video_frames_pipe_endpoint.poll():
                dpg.set_value("VideoTexture",self.video_frames_pipe_endpoint.recv())
        if self.infiriendo:
            #Ajustamos el icono de cargado para que este centrado
            dpg.set_item_width("LeftSpacerInfiriendo",(dpg.get_item_rect_size(self.active_window)[0]-dpg.get_item_rect_size("LoadingIcon")[0]-3)/2)

            self.nuevo_frame = False
            while self.frame_enpoint.poll():
                datos:pd.DataFrame = self.frame_enpoint.recv()
                if type(datos)==int:
                    self.nuevo_frame = True
                    if self.nuevo_frame:
                        dpg.set_value(item="progreso",value=(datos+1)/self.total_frames)
                        dpg.set_value(item="TextProgreso",value=f"{datos +1} fotogramas de {self.total_frames} procesados")
                else:
                    #Limpiamos variables anteriores
                    self.infiriendo=False
                    self.left_moves = 0
                    self.right_moves = 0
                    self.dataset_global_left = None
                    self.dataset_global_right = None
                    #Empezamos a trabajar con los nuevos datos
                    self.raw_data = datos
                    self.listas_movimientos = datos[0]
                    #1. Obtenemos todas las series de datos que vamos a usar para asignar a las gráficas a través del dataframe resultante
                    self.dataset_global_left:pd.DataFrame = datos[1]
                    if len(datos) == 3 : self.dataset_global_right:pd.DataFrame = datos[2]
                    #2. Obtenemos la cantidad de frames del dataframe
                    self.total_frames = len(self.dataset_global_left)
                    #3. Obtenemos la cantidad de movimientos por lado
                    if len(datos[0])==1:
                        fish_number = 1
                        contador_none_izquierda = sum(1 for x in datos[0][0] if x[1] is None)
                        self.left_moves = len(datos[0][0]) - contador_none_izquierda
                    else:
                        fish_number = 2
                        contador_none_izquierda = sum(1 for x in datos[0][0] if x[1] is None)
                        contador_none_derecha = sum(1 for x in datos[0][1] if x[1] is None)
                        self.left_moves = len(datos[0][0]) - contador_none_izquierda
                        self.right_moves = len(datos[0][1]) - contador_none_derecha
                    #Configurar las gráficas
                    self.sub_arrays_processing()
                    self.set_data_graphs()
                    #Configurar textos
                    dpg.set_value("MovimientosIzquierda",f"Numero total de movimientos del pez izquierda:\n{self.left_moves} movimientos")
                    if fish_number == 2: dpg.set_value("MovimientosDerecha",f"Numero total de movimientos del pez derecho:\n{self.right_moves} movimientos")

                    #Configuramos la primera imágen del video
                    self.set_first_image()

                    #Iniciamos el proceso de obtención de imagen del video
                    control_pipe = mp.Pipe(duplex=False) #(Output,Input) de una conexión unidireccional entre el proceso main y el proceso de resultados
                    video_frames_pipe = mp.Pipe(duplex=False) #(Output,Input) de una conexión unidireccional entre el proceso main y el proceso de resultados
                    self.control_pipe_endpoint = control_pipe[1]
                    self.video_frames_pipe_endpoint = video_frames_pipe[0]
                    self.video_controller = Controller(control_pipe[0],video_frames_pipe[1],dpg.get_value('inputText1'))
                    self.video_controller.start()
                    #Cambiamos de ventana
                    self.set_window("DataWindow")


    def set_data_graphs(self):
        eje_frame = np.linspace(start=0,stop=self.total_frames-1,num=self.total_frames) # Creamos el eje de fotogramas
        self.eje_frame = eje_frame
        #Configuramos las gráficas de la Izquierda
        dpg.set_value("Area_Izquierda",[eje_frame, self.dataset_global_left['area'].tolist()])
        dpg.set_value("Centroide_Izquierda",[eje_frame, self.dataset_global_left['centroide_change'].tolist()])
        dpg.set_value("Relacion_Izquierda",[eje_frame, self.dataset_global_left['width_height_relation'].tolist()])
        dpg.set_value("Blur_Izquierda",[eje_frame, self.dataset_global_left['blur'].tolist()])
        dpg.set_value("Area_IzquierdaCompleta",[eje_frame, self.dataset_global_left['area'].tolist()])
        dpg.set_value("Centroide_IzquierdaCompleta",[eje_frame, self.dataset_global_left['centroide_change'].tolist()])
        dpg.set_value("Blur_IzquierdaCompleta",[eje_frame, self.dataset_global_left['blur'].tolist()])
        #Valores de la timeline
        dpg.set_value("ZonasConFrame",[eje_frame,self.real_mov_left,np.zeros(self.real_mov_left.shape)])
        dpg.set_value("ZonasSinFrame",[eje_frame,self.prob_mov_left,np.zeros(self.prob_mov_left.shape)])
        dpg.set_value("ZonasConFrame2",[eje_frame,self.real_mov_right,np.zeros(self.real_mov_right.shape)])
        dpg.set_value("ZonasSinFrame2",[eje_frame,self.prob_mov_right,np.zeros(self.prob_mov_right.shape)])
        dpg.set_value("ClickedStem",[eje_frame,np.full(self.total_frames,-1)])

        if type(self.dataset_global_right)!= type(None):
            #Muestro los items
            dpg.show_item("Area_Derecha")
            dpg.show_item("Centroide_Derecha")
            dpg.show_item("Relacion_Derecha")
            dpg.show_item("Blur_Derecha")
            dpg.show_item("GraficaFinalDerecha")
            dpg.show_item("MovimientosDerecha")
            dpg.show_item("TimeLinePlot2")
            dpg.show_item("AddRight")
            dpg.show_item("DelRight")
            #Los cargo de datos
            dpg.set_value("Area_Derecha",[eje_frame, self.dataset_global_right['area'].tolist()])
            dpg.set_value("Centroide_Derecha",[eje_frame, self.dataset_global_right['centroide_change'].tolist()])
            dpg.set_value("Relacion_Derecha",[eje_frame, self.dataset_global_right['width_height_relation'].tolist()])
            dpg.set_value("Blur_Derecha",[eje_frame, self.dataset_global_right['blur'].tolist()])
            dpg.set_value("Area_DerechaCompleta",[eje_frame, self.dataset_global_right['area'].tolist()])
            dpg.set_value("Centroide_DerechaCompleta",[eje_frame, self.dataset_global_right['centroide_change'].tolist()])
            dpg.set_value("Blur_DerechaCompleta",[eje_frame, self.dataset_global_right['blur'].tolist()])
            dpg.set_value("ClickedStem2",[eje_frame,np.full(self.total_frames,-1)])
        else:
            dpg.hide_item("Area_Derecha")
            dpg.hide_item("Centroide_Derecha")
            dpg.hide_item("Relacion_Derecha")
            dpg.hide_item("Blur_Derecha")
            dpg.hide_item("GraficaFinalDerecha")
            dpg.hide_item("MovimientosDerecha")
            dpg.hide_item("TimeLinePlot2")
            dpg.hide_item("AddRight")
            dpg.hide_item("DelRight")
        #Les pongo limites de zoom-out en x
        dpg.set_axis_limits("area_x",0,self.total_frames)
        dpg.set_axis_limits("centroide_x",0,self.total_frames)
        dpg.set_axis_limits("hw_x",0,self.total_frames)
        dpg.set_axis_limits("blur_x",0,self.total_frames)
        dpg.set_axis_limits("xCompletaIzquierda",0,self.total_frames)
        dpg.set_axis_limits("xCompletaDerecha",0,self.total_frames)
        dpg.set_axis_limits("TimeLineXDerecha",0,self.total_frames)
        dpg.set_axis_limits("TimeLineXIzquierda",0,self.total_frames)
        #AutoFit sobre los axis mvYAxis que utilizamos
        dpg.fit_axis_data("AreaYIzquierda")
        dpg.fit_axis_data("AreaYDerecha")
        dpg.fit_axis_data("CentroideYIzquierda")
        dpg.fit_axis_data("CentroideYDerecha")
        dpg.fit_axis_data("RelacionYI")
        dpg.fit_axis_data("RelacionYD")
        dpg.fit_axis_data("BlurYI")
        dpg.fit_axis_data("BlurYD")
        dpg.fit_axis_data("YAreaI")
        dpg.fit_axis_data("YCentroideI")
        dpg.fit_axis_data("YBlurI")
        dpg.fit_axis_data("YAreaD")
        dpg.fit_axis_data("YCentroideD")
        dpg.fit_axis_data("YBlurD")


    def sub_arrays_processing(self):
        """
            Crea las series
        """
        self.left_frames = []
        self.right_frames = []
        self.prob_mov_left = np.zeros(self.total_frames)
        self.prob_mov_right = np.zeros(self.total_frames)
        self.real_mov_left = np.zeros(self.total_frames)
        self.real_mov_right = np.zeros(self.total_frames)
        for i in self.listas_movimientos[0]:
            if i[1]==None:
                for frame in i[0]:
                    self.prob_mov_left[frame] = 1
            else:
                self.left_frames.append(int(i[1]))
                for frame in i[0]:
                    self.real_mov_left[frame] = 1
        if len(self.listas_movimientos) == 2:
            for i in self.listas_movimientos[1]:
                if i[1]==None:
                    for frame in i[0]:
                        self.prob_mov_right[frame] = 1
                else:
                    self.right_frames.append(int(i[1]))
                    for frame in i[0]:
                        self.real_mov_right[frame] = 1
    
    def set_first_image(self):
        if self.first_image is not None:
            self.actual_image = self.first_image.copy()
            frame_rgb = cv.resize(self.first_image,(720,480))
            frame_rgb = cv.cvtColor(frame_rgb,cv.COLOR_BGR2RGB)
            
            data = frame_rgb.flatten().astype(np.float32)/255
            dpg.set_value("VideoTexture",data)

    def next_frame(self):
        ret, frame = self.cap.read()
        data = None
        if ret:
            self.actual_image = frame.copy()
            frame_rgb = cv.resize(frame,(720,480))
            frame_rgb = cv.cvtColor(frame_rgb,cv.COLOR_BGR2RGB)
            data = frame_rgb.flatten().astype(np.float32)/255
        return data

    def set_window(self,window_name:str):
        """
            Método para cambiar a una ventana específica y esconder el resto
        """
        extra_array = self.window_tags.copy()
        extra_array.pop(extra_array.index(window_name))
        for i in extra_array:
            dpg.hide_item(i)
        dpg.show_item(window_name)
        self.active_window = window_name
        if window_name == "DataWindow":
            self.data_mode = True
        else:
            self.data_mode = False
        dpg.set_primary_window(window_name,True)

    def general_close_callback(self):
        try:
            if self.infiriendo:
                self.data_processor.terminate()
                self.data_processor.close()
            elif self.data_mode:
                self.video_controller.terminate()
                self.video_controller.close()
        except:
            pass
        finally:
            if self.infiriendo:
                self.model.stop_video_inference()
