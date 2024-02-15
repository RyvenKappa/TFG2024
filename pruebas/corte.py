from moviepy.editor import *

clip = VideoFileClip("prueba.MTS")

clip = clip.subclip(27,60)
