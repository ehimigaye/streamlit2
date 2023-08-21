#Commencons par  nous occuper des données à utiliser.
from covid.api import CovId19Data
api = CovId19Data(force=False)

res = api.get_stats()
print(res)

#voyons les données
re = api.get_all_records_by_country()
print(re)

#transformons les données en dataframe .
import pandas as pd
d = pd.DataFrame(re)
print(d.head(5))

#Pour une meilleure utilisation de la données nous allons transposer le dataframe.
D = d.T
print(D.head(5))

#nous allons extraire du dataframe un fichier excel pour pouvoir utiliser le logiciel tableau

import numpy as np
file_name = 'D.xlsx'
D.to_excel(file_name)

#faisons quelques visualisations
import plotly.express as px
fig = px.bar(D, x="label", y="deaths", color="deaths", height=400)
fig.update_layout(title_text="Morts du covid par pays", xaxis_title="pays", yaxis_title="Nombre de morts")


fig1 = px.bar(D, x="label", y="confirmed", color="confirmed", height=400)
fig1.update_layout(title_text= "infectés du covid par pays", xaxis_title="pays", yaxis_title="Nombre de cas confirmés")


import plotly.graph_objects as px
import numpy as np
plot = px.Figure(data=[px.Scatter(
    x = D['confirmed'],
    y = D['deaths'],
    mode = 'markers',)
])
plot.show()
#un nuage de point concernant le nombre de mort et de cas de contamination
#Nous allons essayer de faire  un modele pour predire le nombre de mort en fonction du nombre cas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import metrics
x=D["confirmed"].values[:,np.newaxis]
y=D["deaths"].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.35,random_state=40) #splitting data with test size of 35%
model=LinearRegression()   #build linear regression model
model.fit(x_train,y_train)  #fitting the training data
predicted=model.predict(x_test) #testing our model’s performance
print("MSE", mean_squared_error(y_test,predicted))
print("R squared", metrics.r2_score(y_test,predicted))

plt.scatter(x,y,color="r")
plt.title("Linear Regression")
plt.ylabel("death")
plt.xlabel("confirmed")
plt.plot(x,model.predict(x),color="k")

#Nous avons pris le soin d' exporter les visualisations en format png à partir de jupiter et tableau
import streamlit as st

st.title("CHECKPOINT 2 streamlit")
st.header("PREDICTION covid")
st.subheader("  Moise senghor")
st.subheader(" Data science bootamp")
st.subheader("    Gomycode")

#Essayons de visualiser les données pour une meilleures comprehensions
st.text("visualisons les données ")
from PIL import Image
img1 = Image.open("infectes.png")
st.image(img1, width=800)
img2 = Image.open("morts.png")
st.image(img2, width=800)
img3 = Image.open("pays.png")
st.image(img1, width=800)
img = Image.open("Figure_1.png")
st.image(img, width=800)

#essayons maintenant de prédire le nombre de morts en fonction du nombre d' infectés
nbr = st.number_input("le nombre de cas confirmés()")

st.write("D' apres le modele pour " ,nbr , "cas confirmés  il y aura ", model.predict([[nbr]]), " morts.")




