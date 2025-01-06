# Modelo de Predicción de Ocupación Hotelera en Bariloche

Hola, mi nombre es Silvina y me gustaría presentarte un modelo que he desarrollado para predecir la ocupación hotelera en la ciudad de Bariloche.

## Importancia de la Predicción de Ocupación Hotelera

Prever la ocupación hotelera en Bariloche es crucial por varias razones:

1. Mejora en la planificación de recursos turísticos.
2. Reducción de costos operativos.
3. Optimización de precios.
4. Mejora de la experiencia del turista.
5. Toma de decisiones estratégicas.

## Desarrollo del Modelo

Para construir este modelo, utilizamos datos históricos de ocupación hotelera en Bariloche, obtenidos del sistema de información turística de Argentina, abarcando el período desde 2019 hasta 2023. Además, incluimos el interés en Google Trends para la palabra "Bariloche" como variable exógena.

## Análisis de la Serie Temporal

Analizamos la serie temporal de ocupación hotelera y la descomponemos en tendencia, estacionalidad y residuos. Observamos:

- **Tendencia:** La ocupación tiende al alza, interrumpida en 2020 por la pandemia de COVID-19.
- **Estacionalidad:** La ocupación es más alta en diciembre, enero y febrero durante el verano, así como en julio y agosto durante el invierno.
- **Residuos:** Intentaremos predecir estos residuos utilizando el interés en Google Trends por la palabra "Bariloche".

## Funcionamiento del Modelo

El modelo utiliza como variables de entrada el año y el mes actual, junto con el valor promedio mensual del interés en Google Trends. Luego, predice la ocupación hotelera del mes siguiente.

## Resultados

Obtuvimos excelentes resultados utilizando el XGBoostRegressor y manteniendo la secuencialidad de los datos en los conjuntos de prueba y test para evitar el leakage. Con una optimización de hiperparámetros mediante Optuna, logramos un valor de R2 de 0.87.

## Incorporación de Google Trends

El uso de Google Trends para mejorar las predicciones resultó muy efectivo. Comparando modelos con y sin esta variable, observamos un aumento significativo en el R2, pasando de 0.42 a 0.87.

## Despliegue del Modelo en Streamlit

El modelo se desplegó utilizando Streamlit y la librería pytrends para automatizar la obtención de datos de Google Trends. Puedes encontrar el enlace al modelo en la descripción de este repositorio, donde obtendrás la predicción de ocupación para el próximo mes y un gráfico histórico.

## Conclusiones y Trabajo Futuro

Desarrollamos una herramienta muy útil para el sector hotelero de Bariloche, demostrando el poder predictivo de Google Trends. Como trabajo futuro, planeamos explorar otras configuraciones de modelos y extender este enfoque a otras localidades turísticas.

## Agradecimientos

Gracias por tu interés en este proyecto. Si tienes alguna pregunta o comentario, no dudes en contactarme a través de GitHub. ¡Espero que este modelo sea de utilidad para ti!

prueba





