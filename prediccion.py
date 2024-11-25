import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import streamlit as st

warnings.filterwarnings('ignore', category=FutureWarning)

# Título y descripción para el usuario
st.title("Predicción de Ventas para la compañía PredictSell")
st.markdown("""
Bienvenido a la herramienta de **predicción de ventas** de **PredictSell**. Esta aplicación te permite hacer proyecciones de ventas para diferentes categorías de productos, utilizando un modelo predictivo basado en los datos históricos de ventas. Puedes elegir entre realizar predicciones para un **mes completo** o para un **día específico**.

### ¿Cómo funciona?

1. **Selección de la categoría de producto**: 
   - Elige la categoría de producto para la cual deseas realizar una predicción de ventas. Esto nos permitirá filtrar los datos y realizar un análisis enfocado.

2. **Definición del período de predicción**: 
   - Puedes seleccionar si deseas obtener la predicción por **mes** o por **día**.
   
   - Si eliges **mes**, la aplicación te pedirá que ingreses el año para el cual deseas ver la proyección mensual de ventas. La predicción te mostrará las unidades esperadas para cada mes del año seleccionado.
   
   - Si eliges **día**, la aplicación te solicitará una fecha específica y te mostrará la cantidad estimada de ventas para ese día.

3. **Generación de las predicciones**: 
   - Basado en los datos históricos de ventas, el modelo de predicción ajusta las variables como el mes, el año, el día de la semana, y otros factores, para estimar las ventas futuras.

¡Estamos encantados de ayudarte a tomar decisiones informadas basadas en nuestras predicciones de ventas!
""")

# Cargar el archivo limpio procesado
df = pd.read_csv("Dataset/cleaned_retail.csv")

# Verificar si el DataFrame está vacío
if df.empty:
    st.error("No se encontraron datos para procesar.")
else:
    # Solicitar al usuario la categoría de producto para la predicción
    product_category = st.selectbox("Seleccione la categoría de producto para analizar:", df['Product_Category'].unique())

    # Filtrar por la categoría de producto seleccionada
    product_data = df[df['Product_Category'].str.strip().str.lower() == product_category.strip().lower()]

    if product_data.empty:
        st.error(f"No se encontraron datos para la categoría de producto '{product_category}'.")
    else:
        # Solicitar al usuario si desea predecir por día o por mes
        prediction_type = st.radio("¿Qué tipo de análisis deseas? Define el periodo:", ('Mes', 'Día'))

        if prediction_type.lower() == 'mes':
            # Asegurarse de que la columna 'Date' es de tipo datetime
            product_data.loc[:, 'Date'] = pd.to_datetime(product_data['Date'])
            product_data.set_index('Date', inplace=True)  # Establecer 'Date' como índice

            # Resample por mes y sumar las cantidades
            monthly_sales = product_data.resample('ME').sum()  # Cambié 'M' a 'ME'

            # Verificar que hay datos históricos suficientes
            if monthly_sales.empty:
                st.error(f"No se encontraron datos históricos suficientes para '{product_category}'.")
            else:
                # Agregar características adicionales para el modelo
                monthly_sales['month'] = monthly_sales.index.month
                monthly_sales['year'] = monthly_sales.index.year
                X = monthly_sales[['year', 'month']]
                y = monthly_sales['Quantity']

                # Entrenar el modelo
                model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
                model.fit(X, y)

                # Predecir para el próximo año
                year_to_predict = st.number_input("Ingrese el año para la predicción de ventas detallado:", min_value=2020, max_value=2100, value=2024)
                future_X = pd.DataFrame({
                    'year': [year_to_predict] * 12,
                    'month': np.arange(1, 13)
                })
                forecast = model.predict(future_X)

                # Mostrar las predicciones
                st.subheader(f"Proyección para la categoría {product_category} en el periodo {year_to_predict}:")
                for month, value in zip(future_X['month'], forecast):
                    st.write(f"{month:02d}/{year_to_predict}: {int(value)} unidades")

                # Graficar
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(future_X['month'], forecast, label='Predicciones')
                ax.set_xlabel('Mes')
                ax.set_ylabel('Unidades Vendidas')
                ax.set_title(f'Proyección de Ventas para {product_category} en {year_to_predict}')
                ax.legend()
                st.pyplot(fig)

        elif prediction_type.lower() == 'día':
            # Asegurarse de que la columna 'Date' es de tipo datetime
            product_data.loc[:, 'Date'] = pd.to_datetime(product_data['Date'])
            product_data.set_index('Date', inplace=True)

            # Resample por día y sumar las cantidades
            daily_sales = product_data.resample('D').sum()  # Cambié 'M' a 'D'

            # Verificar que hay datos históricos suficientes
            if daily_sales.empty:
                st.error(f"No se encontraron datos históricos suficientes para '{product_category}'.")
            else:
                # Agregar características adicionales para el modelo
                daily_sales['day'] = daily_sales.index.day
                daily_sales['month'] = daily_sales.index.month
                daily_sales['year'] = daily_sales.index.year
                daily_sales['weekday'] = daily_sales.index.weekday  # Día de la semana (0=Monday, 6=Sunday)
                daily_sales['is_weekend'] = daily_sales['weekday'].isin([5, 6]).astype(int)  # Fin de semana (1=Sí, 0=No)
                daily_sales['day_of_year'] = daily_sales.index.dayofyear  # Día del año para estacionalidad

                # Si se tiene algún tipo de fecha o dia especiafica que se quiera trabjar en un futuro lo podremos agregar en esta parte de cógido
                holidays = ['2024-01-01', '2024-12-25']  
                daily_sales['is_holiday'] = daily_sales.index.strftime('%Y-%m-%d').isin(holidays).astype(int)

                # Variables predictoras (X) y la variable objetivo (y)
                X = daily_sales[['year', 'month', 'day', 'weekday', 'is_weekend', 'day_of_year', 'is_holiday']]
                y = daily_sales['Quantity']

                # Dividir los datos en entrenamiento y prueba
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Entrenar el modelo
                model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
                model.fit(X_train, y_train)

                # Solicitar al usuario una fecha específica para la predicción
                date_to_predict = st.date_input("Ingrese la fecha para la predicción de ventas:", min_value=pd.to_datetime("2020-01-01"))
                date_to_predict = pd.to_datetime(date_to_predict)

                # Crear entrada para el día futuro (con las características de estacionalidad)
                future_X = pd.DataFrame({
                    'year': [date_to_predict.year],
                    'month': [date_to_predict.month],
                    'day': [date_to_predict.day],
                    'weekday': [date_to_predict.weekday()],
                    'is_weekend': [1 if date_to_predict.weekday() in [5, 6] else 0],
                    'day_of_year': [date_to_predict.dayofyear],
                    'is_holiday': [1 if date_to_predict.strftime('%Y-%m-%d') in holidays else 0]
                })

                # Predecir las ventas para el día futuro
                forecast = model.predict(future_X)

                # Mostrar la predicción para el día especificado
                st.subheader(f"Proyección de ventas para {product_category} el día {date_to_predict.strftime('%d/%m/%Y')}:")
                predicted_units = int(forecast[0])  # Convertir la predicción a un número entero

                # Mensaje de salia para la repsuesta de pryección diaria
                st.write(f"Para la fecha indicada se prevé vender un total de {predicted_units} unidades para la categoria {product_category}.")
                st.write("Esta predicción puede ayudar a ajustar el stock disponible para ese día y optimizar la gestión de inventario.")


        else:
            st.error("Opción no válida. Elija 'Mes' o 'Día'.")


 