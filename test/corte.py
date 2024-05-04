from moviepy.editor import *

clip = VideoFileClip("C:/Users/Diego/Documents/Código/TFG2024/resources/videos/prueba.MTS")

clip = clip.subclip(410,486)

clip.write_videofile("targetTestExp2.MTS",fps=25)