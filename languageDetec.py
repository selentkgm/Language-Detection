"""

"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import tkinter as tk

df = pd.read_csv("Language-Detection.csv")
# print(df)
# print(df.columns)
# print(df['Language'].value_counts())

X = df["Text"]
Y = df["Language"]

countV = CountVectorizer()
xs = countV.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(xs, Y, test_size=0.2)

nb = MultinomialNB()
nb.fit(X_train, y_train)

# print(nb.score(X_test, y_test))


def detectLang():
    txt = entry.get()
    print(txt)
    test = countV.transform([txt]).toarray()
    print(nb.predict(test))
    result = nb.predict(test)
    result_label.config(text=f"Detected Language -> {result}")
    entry.delete(0, tk.END)


window = tk.Tk()
window.title("Language Detection")
window.iconbitmap("translate.ico")

canvas = tk.Canvas(window, width=200, height=200)
image = tk.PhotoImage(file="image.png")
canvas.create_image(100, 100, image=image)

window.geometry("300x300")

label = tk.Label(text="Please write something you want.", width=50)
entry = tk.Entry(width=30)

result_label = tk.Label(window, text="Detected Language: ", width=50)

button = tk.Button(window, text='Detect Language', width=15, command=detectLang)

label.pack()
entry.pack()
button.pack()
canvas.pack()
result_label.pack()

window.mainloop()
