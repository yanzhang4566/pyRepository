import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt


def select_file1():
    file1.set(filedialog.askopenfilename())
    return

def select_file2():
    file2.set(filedialog.askopenfilename())
    return

def plot_sq():
    f1 = open(file1.get())
    a=int(f1.readline())
    f1.close()
    f2 = open(file2.get())
    b=int(f2.readline())
    f2.close()

    plt.bar([0,6], [a,b], width = 0.35, color = '#ff0000', lw = 1)
    plt.show()

#window
root= tk.Tk()
root.title('swath sequence plot')
root.geometry('600x200')

file1 = tk.StringVar()
tk.Label(root,text='Uiclog: ', width = 10, height = 2).grid(row = 2, column = 0)
tk.Entry(root,textvariable=file1, width = 70).grid(row = 2, column = 1)
tk.Button(root,text='Select ', command = select_file1).grid(row = 2, column = 2)

file2 = tk.StringVar()
tk.Label(root,text='Galog: ', width = 10, height = 2).grid(row = 3, column = 0)
tk.Entry(root,textvariable=file2, width = 70).grid(row = 3, column = 1)
tk.Button(root,text='Select ', command = select_file2).grid(row = 3, column = 2)

tk.Button(root,text='PLOT ', command = plot_sq).grid(row = 5, column = 0)

root.mainloop()



'''

#window
window= tk.Tk()
window.title('my window')
window.geometry('400x300')
window.mainloop()

#label
var = tk.StringVar()
l = tk.Label(window, textvariable=var, bg='blue', font=('Arial',12),
             width = 15, height = 2)
l.pack()

#entry
e = tk.Entry(window, show=None)
e.pack()

#text
t = tk.Text(window, height = 2)
t.pack()

#button
b = tk.Button(window, text='insert poit', width = 15, height = 2,
              command=insert_p)
b.pack()

#listbox
lb = tk.Listbox(window, listvariable=var2)
lb.pack()

#radiobutton
r = tk.Radiobutton(window, text='Op_A', variable=var, value= 'A',
                    command = print_selection)
r.pack()

#Scale
s = tk.Scale(window, from_ = 5, to_ = 10,
             orient=tk.HORIZONTAL, length=350,
             showvalue=0, tickinterval=1, resolution=0.1,
             command = print_selection)
'''




