import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import tkinter as tk

window = tk.Tk()

title = tk.Label(text="Language Deteciton",
                 foreground="blue",
                 background="black",
                 width=100,
                 height=100)

df = pd.read_csv("Language Detection.csv")
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

text = input("Please write something: ")
test = countV.transform([text]).toarray()
print(nb.predict(test))

window.mainloop()