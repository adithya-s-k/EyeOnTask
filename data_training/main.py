import tkinter as tk 
from data_collection import data_collection
from training import training
from inference import inference
import tkinter.font as font


window = tk.Tk()
window.title("live emoji")
window.geometry("450x140")

frame1 = tk.Frame(window)

btn_font = font.Font(size=15)
btn1 = tk.Button(frame1,fg="red", text="add data",command=data_collection, height=5, width=10)
btn1['font'] = btn_font
btn1.grid(row=0, column=0, padx=(5,5), pady=(10,10))

btn2 = tk.Button(frame1,fg="orange", text="train",command=training, height=5, width=10)
btn2['font'] = btn_font
btn2.grid(row=0, column=1, padx=(5,5), pady=(10,10))

btn3 = tk.Button(frame1,fg="green", text="run",command=inference, height=5, width=10)
btn3['font'] = btn_font
btn3.grid(row=0, column=2, padx=(5,5), pady=(10,10))

frame1.pack()
window.mainloop()