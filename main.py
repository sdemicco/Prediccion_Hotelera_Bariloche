import streamlit as st
import numpy as np
import pandas as pd
import pytrends
import xgboost as xgb
from PIL import Image
from datetime import datetime
from pytrends.request import TrendReq 
import matplotlib.pyplot as plt
import seaborn as sns


background_color = "#EEEEEE"

# Aplicar estilos CSS a través de HTML en Markdown
st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: {background_color};
        }}
    </style>
    """,
    unsafe_allow_html=True
)


# importo el modelo
model = xgb.XGBRegressor()
model.load_model('xgb_model_turismo_5.json')

#importo dataset con los valores reales
valores_reales=pd.read_csv('valores_reales')

mes_actual = datetime.now().month
anio_actual = datetime.now().year

# dicionarios para hacer ajustes de mes y fechas
dic_mes ={
    1: 31,
    2: 28,
    3: 31,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31
}


nombre_dict = {
    1: 'Enero',
    2: 'Febrero',
    3: 'Marzo',
    4: 'Abril',
    5: 'Mayo',
    6: 'Junio',
    7: 'Julio',
    8: 'Agosto',
    9: 'Septiembre',
    10: 'Octubre',
    11: 'Noviembre',
    12: 'Diciembre'
}

st.markdown('<h1 style="text-align: center; color: #172774;">Predicción de Ocupación Hotelera en la Ciudad de Bariloche</h1>', unsafe_allow_html=True)


st.markdown(
    "<p style='color: #172774; font-size: 20px;'>La predicción es calculada utilizando valores históricos de ocupación y el valor de interés de la palabra Bariloche en Google Trends.</p>",
    unsafe_allow_html=True
)


# Intervalo de confianza calculado teniendo en cuenta el error en test. 
intervalo_confianza_test=55566.85471623692


# generacion de dataset con trends de google
pytrends = TrendReq(hl='es-AR', tz=180)
kw_list = ["Bariloche"]
pytrends.build_payload(kw_list, cat= 0 , timeframe= 'today 5-y' )
data = pytrends.interest_over_time()
data = data.reset_index()


data['date'] = pd.to_datetime(data['date'])
data['Mes']=data['date'].dt.month
data['Anio']=data['date'].dt.year
data_agrupado=data.groupby(['Anio','Mes'])['Bariloche'].mean().reset_index()

nuevo_registro = {'Anio': anio_actual, 'Mes': mes_actual + 1 , 'Bariloche': data_agrupado['Bariloche'].iloc[-1]}
df_nuevo_registro = pd.DataFrame([nuevo_registro])


data_agrupado = data_agrupado.sort_values(by=['Anio', 'Mes'])
data_agrupado['valor_mes_anterior'] = data_agrupado['Bariloche'].shift(1)

mascara= data_agrupado['Anio']>2022
data_agrupado = data_agrupado[mascara]


# Agrego el registro para el calculo del mes siguiente
data_agrupado = pd.concat([data_agrupado, df_nuevo_registro], ignore_index=True)

#data_agrupado.to_csv('data_agrupado')

#data_agrupado= pd.read_csv('data_agrupado')

# mascara=(data_agrupado['Anio']==anio_actual)&(data_agrupado['Mes']==mes_actual)
# input_modelo = data_agrupado[mascara]
#input_transformado= pd.DataFrame({
#    'valor_mes_anterior': input_modelo['Bariloche'],
#    'Mes': input_modelo['Mes'],
#    'Año': input_modelo['Anio']
# })
 
 # Genero el dataset de input para el modelo 
input_transformado= pd.DataFrame({
    'valor_mes_anterior': data_agrupado['valor_mes_anterior'],
    'Mes': data_agrupado['Mes'],
    'Año': data_agrupado['Anio']
})

# Genero un dataset con con los datos para hacer el grafico de salida, convirtiendo los valores de pernoctes a ocupacion con la cantidad de plazas totales por los dias del mes.
para_grafico=input_transformado.copy()

para_grafico['salida'] = model.predict(input_transformado)
para_grafico['total_plazas_mes']=(input_transformado['Mes'].replace(dic_mes))*24653
para_grafico['ocupacion_prediccion']=(para_grafico['salida'] /para_grafico['total_plazas_mes'])*100
para_grafico['limite_superior_test'] = para_grafico['ocupacion_prediccion'] + (intervalo_confianza_test/para_grafico['total_plazas_mes'])*100
para_grafico['limite_inferior_test'] = para_grafico['ocupacion_prediccion']- (intervalo_confianza_test/para_grafico['total_plazas_mes'])*100
para_grafico['nombre_mes']=para_grafico['Mes'].replace(nombre_dict)


ocupacion_actual = para_grafico.ocupacion_prediccion[(para_grafico['Mes'] == mes_actual+1) & (para_grafico['Año'] == anio_actual)].values[0]
nombre_mes_actual = para_grafico.nombre_mes[(para_grafico['Mes'] == mes_actual + 1) & (para_grafico['Año'] == anio_actual)].values[0]
ocupacion_actual = round(float(ocupacion_actual), 2)

#st.sidebar.markdown(f"<h1 style='text-align: center; color:#FF0075; font-size: 40px;'>{nombre_mes_actual}</h1>", unsafe_allow_html=True)

#st.sidebar.markdown(f"<div style='border: 1px solid #FF0075; padding: 10px; border-radius: 5px; text-align: center; font-size: 40px;'>{ocupacion_actual}%</div>", unsafe_allow_html=True)

st.markdown(f"<h1 style='text-align: center; color:#FF0075; font-size: 40px;'>Ocupación {nombre_mes_actual}</h1>", unsafe_allow_html=True)
# st.markdown(f"<div style='border: 1px solid #FF0075; padding: 10px; border-radius: 5px; text-align: center; font-size: 40px;'>{ocupacion_actual}%</div>", unsafe_allow_html=True)

st.markdown(
    f"""
    <div style='border: 1px solid {'#FF0075'}; padding: 10px; border-radius: 5px; text-align: center; font-size: 40px; color: #FF0075;'>
        {ocupacion_actual}%
    </div>
    """,
    unsafe_allow_html=True
)




st.markdown("")
st.markdown("")

para_grafico = pd.merge(para_grafico, valores_reales, on=['Año', 'Mes'], how='left')



# Crear el gráfico con matplotlib
fig, ax = plt.subplots(figsize=(10, 6))

# Graficar el valor de salida
ax.plot(para_grafico['Año'].astype(str) + '-' + para_grafico['Mes'].astype(str), para_grafico['ocupacion_prediccion'], label='Predicción', marker='o', linewidth=2, color='#FF0075')
ax.plot(para_grafico['Año'].astype(str) + '-' + para_grafico['Mes'].astype(str), para_grafico['ocupacion_real'], label='Real', marker='x', linewidth=2, color='#172774')

# Graficar los rangos de límite superior e inferior
ax.fill_between(para_grafico['Año'].astype(str) + '-' + para_grafico['Mes'].astype(str), para_grafico['limite_inferior_test'], para_grafico['limite_superior_test'], color='gray', alpha=0.3, label='Intervalo de Confianza (95%)')

# Añadir etiquetas y leyenda
ax.set_xlabel('Año-Mes', fontsize=14)
ax.set_ylabel('% Ocupación Real y Predicción', fontsize=14)
ax.set_title('Predicción de Ocupación Hotelera con Intervalo de Confianza 95%', fontsize=20)
ax.legend()

# Añadir los valores de predicción en cada punto
for i, valor_prediccion in enumerate(para_grafico['ocupacion_prediccion']):
    ax.annotate(f'{valor_prediccion:.2f}%', ((para_grafico['Año'].astype(str) + '-' + para_grafico['Mes'].astype(str))[i], valor_prediccion), textcoords="offset points", xytext=(0, 5), ha='center')

# Rotar las fechas en el eje x
plt.xticks(rotation=90)

# Ajustar el diseño para evitar la superposición
plt.tight_layout()

# Mostrar el gráfico en Streamlit
st.pyplot(fig)
