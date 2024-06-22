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
import pprint

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
        self.prob_mov_left = []
        self.prob_mov_right = []
        self.real_mov_left = []
        self.real_mov_right = []
        self.clicked_left = None
        self.clicked_right = None
        self.new_click = False
        self.eje_frame = np.linspace(start=1,stop=190,num=190) # por ejemplo    
        #Clicked handlers
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
                            dpg.add_input_text(
                                source="config_user",
                                no_spaces=True,
                                width=2000,
                                hint="Introduzca el Usuario:",
                                tag="UsuarioInputText"
                            )
                            dpg.add_button(label="Cargar usuario",callback=self.set_user_callback)#TODO
                        dpg.add_image(texture_tag="texturaGamma",tag="ImagenGamma")
            with dpg.table(tag="MainTable",resizable=False,header_row=False,reorderable=True):
                dpg.add_table_row()
                dpg.add_table_row()
                dpg.add_table_row()
                dpg.add_table_row()
                dpg.add_table_row()
                dpg.add_table_column(width_stretch=True)
                with dpg.table_row():
                    dpg.add_text(default_value="Seleccione el video sobre el que realizar la inferencia:")
                with dpg.table_row():
                    dpg.add_button(label="Seleccionar fichero", callback=lambda:dpg.configure_item("__Explorador",show=True))
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
                        #dpg.add_image(texture_tag="pescadoGirando",tag="TexturaPez")#TODO pescado girando
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
                        texto_carga = dpg.add_text(default_value=f"Vamos por el frame 0 de 0",label="Texto de progreso",tag="TextProgreso")
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
                                        text = dpg.add_text("Numero total de movimientos del pez izquierdo: ",tag="MovimientosIzquierda")
                                        dpg.bind_item_font(text,"SmallFont")
                                        text = dpg.add_text("Numero total de movimientos del pez derecho: ",tag="MovimientosDerecha")
                                        dpg.bind_item_font(text,"SmallFont")
                    with dpg.child_window(no_scrollbar=True,height=-1,border=False):
                        with dpg.child_window(no_scrollbar=True,height=-300,tag="EditarWindow"):
                            with dpg.child_window(no_scrollbar=True,width=-1,height=100,tag="VideoWindow"):
                                dpg.add_text("Hay que conseguir meter aquí la imagen redimensionada respecto a la ventana")
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
                                                dpg.add_button(label="PLAY",width=-1)
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
                                                        dpg.add_shade_series(self.eje_frame,y1=np.ones(190),tag="ZonasSinFrame",parent="y_axis")  # Rojo
                                                        dpg.bind_item_theme(dpg.last_item(),"timeline_green_theme")
                                                        dpg.set_axis_limits("y_axis",0,1)
                                                dpg.bind_item_handler_registry("TimeLinePlot","Handlers")

                                                with dpg.plot(width=-1, height=-1,tag="TimeLinePlot2"):

                                                    # Añadir ejes
                                                    x_axis = dpg.add_plot_axis(dpg.mvXAxis,tag="TimeLineXDerecha")
                                                    with dpg.plot_axis(dpg.mvYAxis,no_tick_labels=True,no_gridlines=True,no_tick_marks=True,tag="y_axis2"):
                                                        # Añadir la serie de sombreado
                                                        dpg.add_shade_series(self.eje_frame,y1=np.ones(190),tag="ZonasConFrame2",parent="y_axis2")  # Rojo
                                                        dpg.bind_item_theme(dpg.last_item(),"timeline_red_theme")
                                                        dpg.set_axis_limits("y_axis2",0,1)
                                                        dpg.add_shade_series(self.eje_frame,y1=np.ones(190),tag="ZonasSinFrame2",parent="y_axis2")  # Rojo
                                                        dpg.bind_item_theme(dpg.last_item(),"timeline_green_theme")
                                                        dpg.set_axis_limits("y_axis2",0,1)

                                                dpg.bind_item_handler_registry("TimeLinePlot2","Handlers")

                                    with dpg.table_row():
                                        with dpg.table(resizable=False,header_row=False,reorderable=True):
                                            dpg.add_table_row()
                                            dpg.add_table_column()
                                            dpg.add_table_column()
                                            dpg.add_table_column()
                                            dpg.add_table_column()
                                            with dpg.table_row():
                                                dpg.add_button(width=-1,label="Añadir a la Izquierda")
                                                dpg.add_button(width=-1,label="Eliminar en la Izquierda")
                                                dpg.add_button(width=-1,label="Añadir a la Derecha")
                                                dpg.add_button(width=-1,label="Eliminar en la Derecha")
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
                                    dpg.add_input_text(hint="Introduzca el nombre del archivo a guardar",width=-1,height=-1,tag="SaveFileNameInput")
                                dpg.add_button(label="Guardar Resultados",width=-1,height=-1,callback=self.save_results_callback)

    def set_user_callback(self,sender,app_data):
        """
            Callback para configurar el usuario de la aplicación y añadir el path de los resultados anteriores
        """
        #TODO
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
        video_path = dpg.get_value('inputText1')
        cap = cv.VideoCapture(video_path)
        self.total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        cap.release()

        results_pipe = mp.Pipe(duplex=False) #(Output,Input) de una conexión unidireccional entre el proceso main y el proceso de resultados
        inference_pipe = mp.Pipe(duplex=False) #Tubería para que el proceso de inferencia le vaya pasando las imágenes al de procesado de resultados

        self.model = model()
        self.data_processor = Data_Processor(inference_pipe[0],results_pipe[1],self.total_frames)
        self.frame_enpoint = results_pipe[0]
        self.model.video_inference(inference_pipe[1],video_path)
        self.data_processor.start()
        self.infiriendo = True
        dpg.set_value(item="TextProgreso",value=f"Vamos por el frame 0 de {self.total_frames}")
        dpg.set_value(item="progreso",value=0.0)


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
        self.set_window("MainWindow")


    def save_results_callback(self,sender,app_data):
        """
            Método que guarda los resultados de la inferencia en un formato común, de manera que se pueda utilizar más tarde
        """
        #TODO hacer que guarde un fichero de verdad, teniendo en cuenta algún problema y haciendo un pop up indicandolo, pero no solucionandolo
        print(self.save_path+dpg.get_value("SaveFileNameInput"))
        pass
    
    def clicked_callback(self,sender,app_data):
        """
            Método para registrar el frame seleccionado
        """
        item_clickeado = dpg.get_item_alias(app_data[1])
        if item_clickeado == "TimeLinePlot":
            self.clicked_left = round(dpg.get_plot_mouse_pos()[0])
            self.new_click = True
            print(f"{self.frame} clicked")
        elif item_clickeado == "TimeLinePlot2":
            self.clicked_right = round(dpg.get_plot_mouse_pos()[0])
            self.new_click = True
            print(f"{self.frame} clicked")



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
                        dpg.set_value(item="TextProgreso",value=f"Vamos por el frame {datos +1} de {self.total_frames}")
                else:
                    #Limpiamos variables anteriores
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
                    dpg.set_value("MovimientosIzquierda",f"Numero total de movimientos del pez derecho: {self.left_moves} movimientos")
                    if fish_number == 2: dpg.set_value("MovimientosDerecha",f"Numero total de movimientos del pez derecho: {self.right_moves} movimientos")
                    self.set_window("DataWindow")


    def set_data_graphs(self):
        eje_frame = np.linspace(start=0,stop=self.total_frames-1,num=self.total_frames) # Creamos el eje de fotogramas
        #Configuramos las gráficas de la Izquierda
        dpg.set_value("Area_Izquierda",[eje_frame, self.dataset_global_left['area'].tolist()])
        dpg.set_value("Centroide_Izquierda",[eje_frame, self.dataset_global_left['centroide_change'].tolist()])
        dpg.set_value("Relacion_Izquierda",[eje_frame, self.dataset_global_left['width_height_relation'].tolist()])
        dpg.set_value("Blur_Izquierda",[eje_frame, self.dataset_global_left['blur'].tolist()])
        dpg.set_value("Area_IzquierdaCompleta",[eje_frame, self.dataset_global_left['area'].tolist()])
        dpg.set_value("Centroide_IzquierdaCompleta",[eje_frame, self.dataset_global_left['centroide_change'].tolist()])
        dpg.set_value("Blur_IzquierdaCompleta",[eje_frame, self.dataset_global_left['blur'].tolist()])
        #Valores de la timeline
        print(f"{len(eje_frame)} y un una señal de {len(self.real_mov_left)}")
        dpg.set_value("ZonasConFrame",[eje_frame,self.real_mov_left,np.zeros(self.real_mov_left.shape)])
        dpg.set_value("ZonasSinFrame",[eje_frame,self.prob_mov_left,np.zeros(self.prob_mov_left.shape)])
        dpg.set_value("ZonasConFrame2",[eje_frame,self.real_mov_right,np.zeros(self.real_mov_right.shape)])
        dpg.set_value("ZonasSinFrame2",[eje_frame,self.prob_mov_right,np.zeros(self.prob_mov_right.shape)])

        if type(self.dataset_global_right)!= type(None):
            #Muestro los items
            dpg.show_item("Area_Derecha")
            dpg.show_item("Centroide_Derecha")
            dpg.show_item("Relacion_Derecha")
            dpg.show_item("Blur_Derecha")
            dpg.show_item("GraficaFinalDerecha")
            dpg.show_item("MovimientosDerecha")
            dpg.show_item("TimeLinePlot2")
            #Los cargo de datos
            dpg.set_value("Area_Derecha",[eje_frame, self.dataset_global_right['area'].tolist()])
            dpg.set_value("Centroide_Derecha",[eje_frame, self.dataset_global_right['centroide_change'].tolist()])
            dpg.set_value("Relacion_Derecha",[eje_frame, self.dataset_global_right['width_height_relation'].tolist()])
            dpg.set_value("Blur_Derecha",[eje_frame, self.dataset_global_right['blur'].tolist()])
            dpg.set_value("Area_DerechaCompleta",[eje_frame, self.dataset_global_right['area'].tolist()])
            dpg.set_value("Centroide_DerechaCompleta",[eje_frame, self.dataset_global_right['centroide_change'].tolist()])
            dpg.set_value("Blur_DerechaCompleta",[eje_frame, self.dataset_global_right['blur'].tolist()])
        else:
            dpg.hide_item("Area_Derecha")
            dpg.hide_item("Centroide_Derecha")
            dpg.hide_item("Relacion_Derecha")
            dpg.hide_item("Blur_Derecha")
            dpg.hide_item("GraficaFinalDerecha")
            dpg.hide_item("MovimientosDerecha")
            dpg.hide_item("TimeLinePlot2")
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
        self.prob_mov_left = np.zeros(self.total_frames)
        self.prob_mov_right = np.zeros(self.total_frames)
        self.real_mov_left = np.zeros(self.total_frames)
        self.real_mov_right = np.zeros(self.total_frames)
        for i in self.listas_movimientos[0]:
            if i[1]==None:
                for frame in i[0]:
                    self.prob_mov_left[frame] = 1
            else:
                for frame in i[0]:
                    self.real_mov_left[frame] = 1
        if len(self.listas_movimientos) == 2:
            for i in self.listas_movimientos[1]:
                if i[1]==None:
                    for frame in i[0]:
                        self.prob_mov_right[frame] = 1
                else:
                    for frame in i[0]:
                        self.real_mov_right[frame] = 1
                
            



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