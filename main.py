import tkinter as tk 
from data_collection import data_collection
from data_collection import data_collection_screen
from data_collection import data_collection_not_screen
from training import training
from inference import inference
from inference import get_prediction
from blink import blinkDetecton
import tkinter.font as font

window = tk.Tk()
window.title("live emoji")
window.geometry("450x300")

frame1 = tk.Frame(window)

btn_font = font.Font(size=10)
btn1 = tk.Button(frame1,fg="red", text="add",command=data_collection, height=5, width=10)
btn1['font'] = btn_font
btn1.grid(row=0, column=0, padx=(5,5), pady=(2,2))

btn2 = tk.Button(frame1,fg="orange", text="train",command=training, height=5, width=10)
btn2['font'] = btn_font
btn2.grid(row=0, column=1, padx=(5,5), pady=(2,2))

btn3 = tk.Button(frame1,fg="green", text="run",command=inference, height=5, width=10)
btn3['font'] = btn_font
btn3.grid(row=0, column=2, padx=(5,5), pady=(2,2))

btn4 = tk.Button(frame1,fg="green", text="predict",command=get_prediction, height=5, width=10)
btn4['font'] = btn_font
btn4.grid(row=1, column=2, padx=(5,5), pady=(2,2))

btn4 = tk.Button(frame1,fg="blue", text="blink",command=blinkDetecton, height=5, width=10)
btn4['font'] = btn_font
btn4.grid(row=0, column=3, padx=(5,5), pady=(2,2))

frame1.pack()
window.mainloop()