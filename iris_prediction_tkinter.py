import pandas as pd
import numpy as np
from tkinter import *
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("iris.csv")

raiz = Tk()

raiz.title("Prediccion de especie de iris")

raiz.resizable(0, 0)

raiz.geometry("300x400")

Label(raiz, text="Petal Length", bg="blue", fg="white").place(x=50, y=50)
Label(raiz, text="Petal Width", bg="blue", fg="white").place(x=50, y=100)
Label(raiz, text="Sepal Length", bg="blue", fg="white").place(x=50, y=150)
Label(raiz, text="Sepal Width", bg="blue", fg="white").place(x=50, y=200)
textbox = Text(raiz, height=3, width=20,)
textbox.place(x=60, y=300)

input = StringVar()
input1 = StringVar()
input2 = StringVar()
input3 = StringVar()
result = StringVar()

e1 = Entry(raiz, font=1, textvariable=input, width=10).place(x=150, y=50)
e2 = Entry(raiz, font=1, textvariable=input1, width=10).place(x=150, y=100)
e3 = Entry(raiz, font=1, textvariable=input2, width=10).place(x=150, y=150)
e4 = Entry(raiz, font=1, textvariable=input3, width=10).place(x=150, y=200)


def entryClear():
    input.set("")
    input1.set("")
    input2.set("")
    input3.set("")
    result.set("")
    textbox.delete(1.0, END)


def getpredict():
    lst = [float(input.get()), float(input1.get()), float(input2.get()),
           float(input3.get())]
    x = df.iloc[:, :4].values
    y = df.iloc[:, 4:5].values
    X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.80,
                                                        random_state=0)
    eg = np.array(lst)
    eg = eg.reshape(1, -1)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(eg)
    textbox.insert(END, y_pred)


Button(raiz, text="Clear", bg="red", fg="white",
       command=entryClear).place(x=150, y=250)

Button(raiz, text="Predecir", bg="green", fg="white",
       command=getpredict).place(x=50, y=250)

raiz.mainloop()
