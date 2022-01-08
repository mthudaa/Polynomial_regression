#Install terlebih dahulu library melalui terminal dengan perintah 'pip install numpy pandas openpyxl plotly scikit-learn tensorflow xlsxwriter'
#gunakan versi python 3.8
import xlsxwriter
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#mencari rumus pola data
def format_coefs(coefs):
    equation_list = [f"{coef}x^{i}" for i, coef in enumerate(coefs)]
    equation = " + ".join(equation_list)

    replace_map = {"x^0": "", "x^1": "x", '+ -': '- ', '^':'**', 'x':'*x'}
    for old, new in replace_map.items():
        equation = equation.replace(old, new)

    return equation

df = pd.read_excel(r'direktori path') #ganti direktori path pada sintaks ini sesuai dengan file path yang akan di prediksi
X = df.durasi.values.reshape(-1, 1) #ganti pada sintaks df.durasi dengan df.(nama kolom x pada data yang akan di prediksi)
x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

fig = px.scatter(df, x='durasi', y='upah', opacity=0.65) #ganti x dan y dengan kolom pada data yang akan di prediksi
degree = 7
poly = PolynomialFeatures(degree)
poly.fit(X)
X_poly = poly.transform(X)
x_range_poly = poly.transform(x_range)

model = LinearRegression(fit_intercept=False)
model.fit(X_poly, df.upah) #ganti pada sintaks df.upah dengan df.(nama kolom y pada data yang akan di prediksi)
y_poly = model.predict(x_range_poly)

equation = format_coefs(model.coef_.round(2))
fig.add_traces(go.Scatter(x=x_range.squeeze(), y=y_poly, name=equation))
fig.show()

#prediksi gaji lembur dalam file excel
workbook = xlsxwriter.Workbook('gaji lembur.xlsx')
worksheet = workbook.add_worksheet()
worksheet.write(0,0, 'durasi')
worksheet.write(0,1, 'upah lembur')
lembur = np.array([11,12,13,14,15])
for i in range(len(lembur)):
    x=lembur[i]
    worksheet.write(i+1, 0, lembur[i])
    worksheet.write(i+1, 1, eval(equation))
workbook.close()
#file excel otomatis tergenerate
