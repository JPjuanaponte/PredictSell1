import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Cargar el archivo CSV en pandas
df = pd.read_csv("ruta_al_archivo/retail_sales_dataset.csv") 

def predict_sales_with_gb(df):
    # Convertir la columna 'Date' a datetime y 'Quantity' a tipo numérico
    df = df.withColumnRenamed('Date', 'ds').withColumnRenamed('Quantity', 'y')
    df_pd = df.toPandas()
    df_pd['ds'] = pd.to_datetime(df_pd['ds'])
    df_pd['y'] = pd.to_numeric(df_pd['y'], errors='coerce')  # Convertir Quantity a numérico (maneja valores no válidos)

    # Crear la interfaz de usuario con Streamlit
    st.title("Predicción de Ventas con Gradient Boosting")
    
    # Solicitar al usuario la categoría de producto para la predicción
    product_category = st.selectbox("Seleccione la categoría de producto:", df_pd['Product_Category'].unique())

    # Filtrar por la categoría de producto seleccionada
    product_data = df_pd[df_pd['Product_Category'] == product_category]

    if product_data.empty:
        st.error(f"No se encontraron datos para la categoría de producto '{product_category}'.")
        return

    # Solicitar al usuario si desea predecir por día o por mes
    prediction_type = st.radio("¿Desea predecir por día o por mes?", ("Mes", "Día"))

    if prediction_type == 'Mes':
        # Agrupar las ventas por mes y sumar las unidades vendidas
        product_data.set_index('ds', inplace=True)
        monthly_sales = product_data.resample('M').sum(numeric_only=True)

        # Filtrar los datos solo hasta el año 2023 para el entrenamiento
        historical_data = monthly_sales[monthly_sales.index.year == 2023]

        if historical_data.empty:
            st.error(f"No se encontraron datos para el año 2023 y la categoría de producto '{product_category}'.")
            return

        # Crear una copia del DataFrame antes de modificarlo
        historical_data = historical_data.copy()  # Crear una copia

        # Agregar características adicionales para el modelo (Mes y Año)
        historical_data['month'] = historical_data.index.month
        historical_data['year'] = historical_data.index.year

        # Definir las variables predictoras (X) y la variable objetivo (y)
        X = historical_data[['year', 'month']]  # Año y mes como características
        y = historical_data['y']  # Unidades vendidas como objetivo

        # Crear y entrenar el modelo Gradient Boosting Regressor
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        model.fit(X, y)

        # Hacer predicciones para los 12 meses del año que el usuario ha ingresado
        year_to_predict = st.number_input("Ingrese el año para la predicción de ventas (ejemplo: 2024):", min_value=2020, max_value=2030, value=2024)
        
        future_X = pd.DataFrame({
            'year': [year_to_predict] * 12,  # El año para todos los meses
            'month': np.arange(1, 13)   # Los meses del año (de 1 a 12)
        })

        # Predicciones del modelo
        forecast = model.predict(future_X)

        # Crear un DataFrame con las predicciones
        forecast_df = pd.DataFrame(forecast, index=pd.date_range(f'{year_to_predict}-01-01', periods=12, freq='M'), columns=['forecast'])

        # Mostrar las predicciones de ventas
        st.write(f"Tendencia de ventas proyectada para la categoría de producto '{product_category}' en el año {year_to_predict}")
        for date, forecast_value in forecast_df.iterrows():
            st.write(f"En {date.strftime('%B %Y')}, la proyección de ventas será de aproximadamente {int(forecast_value['forecast'])} unidades.")

        # Graficar los resultados (mostrando solo las predicciones para el año ingresado)
        plt.figure(figsize=(10, 6))
        plt.plot(forecast_df.index, forecast_df['forecast'], label=f'Proyección de ventas {year_to_predict}', color='orange')
        plt.title(f"Tendencia de ventas proyectada para la categoría de producto '{product_category}' en el año {year_to_predict}")
        plt.xlabel("Mes")
        plt.ylabel("Unidades Vendidas")
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True)
        st.pyplot()

    elif prediction_type == 'Día':
        # Agrupar las ventas por día y sumar las unidades vendidas
        product_data.set_index('ds', inplace=True)
        daily_sales = product_data.resample('D').sum(numeric_only=True)

        # Agregar características adicionales para el modelo (Día, Mes y Año)
        daily_sales['year'] = daily_sales.index.year
        daily_sales['month'] = daily_sales.index.month
        daily_sales['day'] = daily_sales.index.day

        # Definir las variables predictoras (X) y la variable objetivo (y)
        X = daily_sales[['year', 'month', 'day']]  # Año, mes y día como características
        y = daily_sales['y']  # Unidades vendidas como objetivo

        # Entrenamiento y división de datos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Crear y entrenar el modelo Gradient Boosting Regressor
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        model.fit(X_train, y_train)

        # Solicitar al usuario la fecha para la predicción
        year_input = st.number_input("Ingrese el año para la predicción (por ejemplo, 2024):", min_value=2020, max_value=2030, value=2024)
        month_input = st.number_input("Ingrese el mes para la predicción (por ejemplo, 11 para noviembre):", min_value=1, max_value=12, value=11)
        day_input = st.number_input("Ingrese el día para la predicción (por ejemplo, 24):", min_value=1, max_value=31, value=24)

        # Crear el DataFrame de entrada para la predicción
        future_X = pd.DataFrame({
            'year': [year_input],
            'month': [month_input],
            'day': [day_input]
        })

        # Predicciones del modelo
        forecast = model.predict(future_X)

        # Mostrar la predicción para el día seleccionado
        st.write(f"Para la fecha {day_input}/{month_input}/{year_input}, la proyección de ventas para la categoría '{product_category}' es de aproximadamente {int(forecast[0])} unidades.")
    else:
        st.error("Opción inválida. Elija 'Día' o 'Mes'.")

# Ejecutar la función
predict_sales_with_gb(df)
