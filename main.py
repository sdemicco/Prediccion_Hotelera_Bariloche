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


background_color = "#E4F9FF"

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

st.markdown('<h1 style="text-align: center; color: #0FABBC;">Predicción de Ocupación Hotelera en la Ciudad de Bariloche</h1>', unsafe_allow_html=True)

#st.markdown( 'La predicción es calculada utilizando valores históricos de ocupación y el valor de interes de la palabra Bariloche obtenido de GoogleTrends del mes anterior.' )
st.markdown(
    "<p style='color: #0FABBC; font-size: 20px;'>La predicción es calculada utilizando valores históricos de ocupación y el valor de interés de la palabra Bariloche obtenido de Google Trends.</p>",
    unsafe_allow_html=True
)


# Intervalo de confianza calculado teniendo en cuenta el error en test. 
intervalo_confianza_test=55566.85471623692

data_agrupado= pd.read_csv('data_agrupado')

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

#st.sidebar.markdown(f"<h1 style='text-align: center; color:#893395; font-size: 40px;'>{nombre_mes_actual}</h1>", unsafe_allow_html=True)

#st.sidebar.markdown(f"<div style='border: 1px solid #893395; padding: 10px; border-radius: 5px; text-align: center; font-size: 40px;'>{ocupacion_actual}%</div>", unsafe_allow_html=True)

st.markdown(f"<h1 style='text-align: center; color:#0FABBC; font-size: 40px;'>Ocupación {nombre_mes_actual}</h1>", unsafe_allow_html=True)
# st.markdown(f"<div style='border: 1px solid #F0F3FF; padding: 10px; border-radius: 5px; text-align: center; font-size: 40px;'>{ocupacion_actual}%</div>", unsafe_allow_html=True)

st.markdown(
    f"""
    <div style='border: 1px solid {'#12CAD6'}; padding: 10px; border-radius: 5px; text-align: center; font-size: 40px; color: #12CAD6;'>
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
ax.plot(para_grafico['Año'].astype(str) + '-' + para_grafico['Mes'].astype(str), para_grafico['ocupacion_prediccion'], label='Predicción', marker='o', linewidth=2, color='#12CAD6')
ax.plot(para_grafico['Año'].astype(str) + '-' + para_grafico['Mes'].astype(str), para_grafico['ocupacion_real'], label='Real', marker='x', linewidth=2, color='#FA163F')

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
