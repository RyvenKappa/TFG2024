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

class Manager():
    """
    Clase que implementa la funcionalidad de las diferentes ventanas de la aplicación
    """
    def __init__(self) -> None:
        self.window_tags=[]
        self.infiriendo = False
        self.raw_data = None
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
                                        dpg.add_spacer()
                                        dpg.add_loading_indicator(radius=7)
                                        dpg.add_spacer()
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
                        with dpg.table(tag="GraphTable",resizable=True,header_row=False,reorderable=True):
                            dpg.add_table_row()
                            dpg.add_table_row()
                            dpg.add_table_column()
                            with dpg.table_row():
                                with dpg.child_window(no_scrollbar=True,height=-300):
                                    with dpg.tab_bar():
                                        with dpg.tab(label="Área"):
                                            dpg.add_text("Gráfica del área")
                                        with dpg.tab(label="Cambio del centroide"):
                                            dpg.add_text("Gráfica del cambio del centroide")
                                        with dpg.tab(label="Cambio de la relación altura ancho"):
                                            dpg.add_text("Gráfica del cambio de la relación altura ancho")
                                        with dpg.tab(label="Cambio del blur pez izquierda"):
                                            dpg.add_text("Gráfica del cambio del blur en pez izquierda")
                                        with dpg.tab(label="Cambio del blur pez derecha"):
                                            dpg.add_text("Gráfica del cambio del blur en pez derecha")
                            with dpg.table_row():
                                dpg.add_text(default_value=f"Zona de gráfica total")
                    with dpg.child_window(no_scrollbar=True,height=-1):
                        dpg.add_text(default_value=f"Zona en la que iran las gráficas")

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

        
    




    def update(self):
        """
            Método que sirve para la actualización de las variables temporales y la comunicación con los procesos de inferencia.
            Además sirve para la actualización de texturas
        """
        if self.infiriendo:
            self.nuevo_frame = False
            while self.frame_enpoint.poll():
                datos:pd.DataFrame = self.frame_enpoint.recv()
                if type(datos)==int:
                    self.nuevo_frame = True
                    if self.nuevo_frame:
                        dpg.set_value(item="progreso",value=(datos+1)/self.total_frames)
                        dpg.set_value(item="TextProgreso",value=f"Vamos por el frame {datos +1} de {self.total_frames}")
                else:
                    print(f"He obtenido unos datos tipo: {type(datos)} y longitud {len(datos)}")
                    print(datos[0])
                    print(datos[1])
                    self.raw_data = datos
                    self.set_window("DataWindow")

    def set_window(self,window_name:str):
        """
            Método para cambiar a una ventana específica y esconder el resto
        """
        extra_array = self.window_tags.copy()
        extra_array.pop(extra_array.index(window_name))
        for i in extra_array:
            dpg.hide_item(i)
        dpg.show_item(window_name)
        dpg.set_primary_window(window_name,True)