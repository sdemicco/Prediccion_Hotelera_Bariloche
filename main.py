import streamlit as st
import numpy as np
import pandas as pd
import pytrends
import xgboost as xgb
from PIL import Image
from datetime import datetime
from pytrends.request import TrendReq 
import matplotlib.pyplot as plt

model = xgb.XGBRegressor()
model.load_model('xgb_model_turismo_3.json')

valores_reales=pd.read_csv('valores_reales')

mes_actual = datetime.now().month
anio_actual = datetime.now().year


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
    1: 'enero',
    2: 'febrero',
    3: 'marzo',
    4: 'abril',
    5: 'mayo',
    6: 'junio',
    7: 'julio',
    8: 'agosto',
    9: 'septiembre',
    10: 'octubre',
    11: 'noviembre',
    12: 'diciembre'
}

st.title('Predicción de Ocupación Hotelera en la Ciudad de Bariloche')


intervalo_confianza_test=50033.69798816929

# generacion de dataset con trends de google
pytrends = TrendReq(hl='es-AR', tz=180)
kw_list = [ "Alojamiento Bariloche" ]
pytrends.build_payload(kw_list, cat= 0 , timeframe= 'today 5-y' )
data = pytrends.interest_over_time()
data = data.reset_index()
data['date'] = pd.to_datetime(data['date'])
data['Mes']=data['date'].dt.month
data['Anio']=data['date'].dt.year
data_agrupado=data.groupby(['Anio','Mes'])['Alojamiento Bariloche'].mean().reset_index()

nuevo_registro = {'Anio': anio_actual, 'Mes': mes_actual + 1 , 'Alojamiento Bariloche': data_agrupado['Alojamiento Bariloche'].iloc[-1]}
df_nuevo_registro = pd.DataFrame([nuevo_registro])






data_agrupado = data_agrupado.sort_values(by=['Anio', 'Mes'])
data_agrupado['valor_mes_anterior'] = data_agrupado['Alojamiento Bariloche'].shift(1)

mascara= data_agrupado['Anio']>2022
data_agrupado = data_agrupado[mascara]

mes_actual = datetime.now().month
anio_actual = datetime.now().year

data_agrupado = pd.concat([data_agrupado, df_nuevo_registro], ignore_index=True)



# mascara=(data_agrupado['Anio']==anio_actual)&(data_agrupado['Mes']==mes_actual)
# input_modelo = data_agrupado[mascara]
#input_transformado= pd.DataFrame({
#    'valor_mes_anterior': input_modelo['Alojamiento Bariloche'],
#    'Mes': input_modelo['Mes'],
#    'Año': input_modelo['Anio']
# })

input_transformado= pd.DataFrame({
    'valor_mes_anterior': data_agrupado['valor_mes_anterior'],
    'Mes': data_agrupado['Mes'],
    'Año': data_agrupado['Anio']
})

para_grafico=input_transformado.copy()


para_grafico['salida'] = model.predict(input_transformado)
para_grafico['total_plazas_mes']=(input_transformado['Mes'].replace(dic_mes))*24653
para_grafico['ocupacion_prediccion']=(para_grafico['salida'] /para_grafico['total_plazas_mes'])*100
para_grafico['limite_superior_test'] = para_grafico['ocupacion_prediccion'] + (intervalo_confianza_test/para_grafico['total_plazas_mes'])*100
para_grafico['limite_inferior_test'] = para_grafico['ocupacion_prediccion']- (intervalo_confianza_test/para_grafico['total_plazas_mes'])*100
para_grafico['nombre_mes']=para_grafico['Mes'].replace(nombre_dict)


ocupacion_actual = para_grafico.ocupacion_prediccion[(para_grafico['Mes'] == mes_actual) & (para_grafico['Año'] == anio_actual)].values[0]
nombre_mes_actual = para_grafico.nombre_mes[(para_grafico['Mes'] == mes_actual + 1) & (para_grafico['Año'] == anio_actual)].values[0]
ocupacion_actual = round(float(ocupacion_actual), 2)

st.sidebar.markdown(f"<h1 style='text-align: center; color:#893395; font-size: 40px;'>{nombre_mes_actual}</h1>", unsafe_allow_html=True)

st.sidebar.markdown(f"<div style='border: 1px solid #893395; padding: 10px; border-radius: 5px; text-align: center; font-size: 40px;'>{ocupacion_actual}%</div>", unsafe_allow_html=True)



para_grafico = pd.merge(para_grafico, valores_reales, on=['Año', 'Mes'], how='left')


# Crear el gráfico con matplotlib
fig, ax = plt.subplots(figsize=(10, 6))

# Graficar el valor de salida
ax.plot(para_grafico['Año'].astype(str) + '-' + para_grafico['Mes'].astype(str), para_grafico['ocupacion_prediccion'], label='Predicción', marker='o')
ax.plot(para_grafico['Año'].astype(str) + '-' + para_grafico['Mes'].astype(str), para_grafico['ocupacion_real'], label='Real', marker='x')

# Graficar los rangos de límite superior e inferior
ax.fill_between(para_grafico['Año'].astype(str) + '-' + para_grafico['Mes'].astype(str), para_grafico['limite_inferior_test'], para_grafico['limite_superior_test'], color='gray', alpha=0.3, label='Intervalo de Confianza (95%)')

# Añadir etiquetas y leyenda
ax.set_xlabel('Año-Mes')
ax.set_ylabel('% Ocupación Real y Predicción')
ax.set_title('Predicción de Ocupación Hotelera con Intervalo de Confianza 95%',fontsize=20)
ax.legend()

# Añadir los valores de predicción en cada punto
for i, valor_prediccion in enumerate(para_grafico['ocupacion_prediccion']):
    ax.annotate(f'{valor_prediccion:.2f}%', ((para_grafico['Año'].astype(str) + '-' + para_grafico['Mes'].astype(str))[i], valor_prediccion), textcoords="offset points", xytext=(0,5), ha='center')

# Ajustar el diseño para evitar la superposición
plt.tight_layout()

# Mostrar el gráfico en Streamlit
st.pyplot(fig)






st.table(input_transformado)