"""
Modulo que contiene las diferentes ventanas de la aplicación, incluyendo las transiciones entre las mismas
@author: Diego Aceituno Seoane
@Mail: diego.aceituno@alumnos.upm.es
"""
import dearpygui.dearpygui as dpg
from yolo import model
import multiprocessing as mp
from results_module import Data_Processor

class Manager():
    """
    Clase que implementa la funcionalidad de las diferentes ventanas de la aplicación
    """
    def __init__(self) -> None:
        self.infiriendo = False
        self.start_screen()

    def start_screen(self) -> None:
        with dpg.window(tag="MainWindow",no_scrollbar=True,autosize=True):
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
                    b_seleccionar = dpg.add_button(label="Seleccionar fichero", callback=lambda:dpg.configure_item("__Explorador",show=True))
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
        results_pipe = mp.Pipe(duplex=False) #(Output,Input) de una conexión unidireccional entre el proceso main y el proceso de resultados
        inference_pipe = mp.Pipe(duplex=False) #Tubería para que el proceso de inferencia le vaya pasando las imágenes al de procesado de resultados

        self.model = model()
        self.data_processor = Data_Processor(inference_pipe[0])
        self.model.video_inference(inference_pipe[1],dpg.get_value('inputText1'))
        self.data_processor.start()
        self.infiriendo = True


        #Cambio de pantalla
        dpg.delete_item("MainWindow")
        self.start_loading_screen()


    def start_loading_screen(self):

        #Creamos la ventana de carga
        with dpg.window(tag="LoadingWindow",no_scrollbar=True,autosize=True):
            dpg.set_primary_window("LoadingWindow",True)
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
                        dpg.add_progress_bar(label="Infiriendo...", tag="progreso",default_value=0.6,width=-1)
                        dpg.add_spacer()
                    with dpg.table_row():
                        dpg.add_spacer()
                        texto_carga = dpg.add_text(default_value="Vamos por el frame 300 de 350",label="Texto de progreso")
                        dpg.bind_item_font(texto_carga,"MidFont")
                        dpg.add_spacer()
            boton_cancelar = dpg.add_button(width=-1,height=150,label="Cancelar Inferencia",show=True,tag="BotonCancelarInferencia",callback=self.cancelar_callback)
            dpg.bind_item_font(boton_cancelar,"MidFont")
        pass


    def cancelar_callback(self,sender,app_data):
        """
            Callback para cancelar la inferencia
        """
        pass

        
    




    def update(self):
        """
            Método que sirve para la actualización de las variables temporales y la comunicación con los procesos de inferencia.
            Además sirve para la actualización de texturas
        """
        #TODO Ver cuantos frames quedan si estamos infiriendo

        pass