#Install terlebih dahulu library melalui terminal dengan perintah 'pip install numpy pandas openpyxl plotly scikit-learn tensorflow'
#gunakan versi python 3.8
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def format_coefs(coefs):
    equation_list = [f"{coef}x^{i}" for i, coef in enumerate(coefs)]
    equation = "$" +  " + ".join(equation_list) + "$"

    replace_map = {"x^0": "", "x^1": "x", '+ -': '- '}
    for old, new in replace_map.items():
        equation = equation.replace(old, new)

    return equation

directory = r'directory' #ganti dengan path file excel
kolom_x = 'x' #ganti dengan nama kolom data x
kolom_y = 'y' #ganti dengan nama kolom data y

df = pd.read_excel(directory)
X = df.abc.values.reshape(-1, 1) #ganti abc dengan nama kolom data x (tanpa tanda petik)
x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

fig = px.scatter(df, x=str(kolom_x), y=str(kolom_y), opacity=0.65)
for degree in [1, 2, 3, 4, 5]: #derajat bisa ditambahkan sesuka hati kalian, tinggal nambahin angka 6 dst
    poly = PolynomialFeatures(degree)
    poly.fit(X)
    X_poly = poly.transform(X)
    x_range_poly = poly.transform(x_range)

    model = LinearRegression(fit_intercept=False)
    model.fit(X_poly, df.abc) #ganti abc dengan nama kolom data y (tanpa tanda petik)
    y_poly = model.predict(x_range_poly)

    equation = format_coefs(model.coef_.round(2))
    fig.add_traces(go.Scatter(x=x_range.squeeze(), y=y_poly, name=equation))
fig.show()
