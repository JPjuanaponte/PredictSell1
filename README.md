# ```Analisis ventas compañia PredictSell``` 

## 🔷Ingreso plataforma tendencias de ventas  en la siguiente imagen:
<div>
    <div align='center'>
    <a href="https://predictsell1-d78gsguhpcrleyito2buua.streamlit.app/" target="_blank" target="_blank">
          <img  src="Images\image_ref.jpg" height=200/>
       </a>
    </div>
</div>
</center>

# Proyecto de Predicción de Ventas Futuras

Este proyecto tiene como objetivo predecir las ventas futuras utilizando un modelo de Machine Learning basado en datos históricos de transacciones. El flujo de trabajo está organizado en varias etapas clave, desde la configuración inicial hasta la implementación final del modelo de predicción. A continuación, se detallan todos los pasos que se siguieron.

<p align="center">
<img src="Images\flujo_ trabajo.jpg"  height=400>
</p>
</center>

## 1. **Configuración Inicial en Azure**

Para comenzar, se configuraron los recursos en **Azure**:
- Se creó un **grupo de recursos** llamado `proyecto-azure`.
- Se configuró una **cuenta de almacenamiento** denominada `storagepredictsell` para almacenar los datos.
- Se estableció un **servicio de Azure Databricks** (`proyect_sales`), que permitió montar un cluster para procesar los datos de manera eficiente.

## 2. **Conexión y Obtención de Datos de Kaggle (API)**

Una vez configurado el entorno, se conectó **Databricks** a la API de Kaggle para descargar un dataset de ventas. Este dataset contiene información sobre transacciones, incluyendo el ID de la transacción, la fecha, el ID del cliente, el género, la edad, la categoría del producto, la cantidad, el precio unitario y el monto total.

- Se configuró un **notebook** en **Databricks** para importar y almacenar los datos descargados en un formato adecuado para su análisis.

## 3. **Transformación y Almacenamiento en Tablas Delta**

Se utilizó **PySpark** para realizar la limpieza de los datos (eliminación de duplicados, manejo de valores nulos y conversión de tipos de datos). Posteriormente, los datos fueron almacenados en una **tabla Delta**. 

## 4. **Automatización del Proceso de Carga y Actualización**

Para mantener los datos siempre actualizados, se automatizó el proceso de carga:
- Se configuró una **tarea programada** para actualizar la tabla Delta de manera diaria, eliminando los datos antiguos para evitar sobrescrituras.
- Se generaron archivos **CSV** por fecha (nombrados como `proyecto_yyyy-MM-dd.csv`) para almacenar los datos de manera granular.

## 5. **Análisis Exploratorio de Datos (EDA)**

Se realizó un análisis exploratorio de los datos utilizando **pandas**, **seaborn**, **matplotlib** y **pyspark**. Los resultados fueron los siguientes:

- El **producto** con mayores ganancias fue la categoría **Electronics**, seguida por **Clothing** y, por último, **Beauty**.
- El **día de la semana** con más ganancias fue el **domingo**, seguido por el **sábado**, mientras que el **viernes** fue el día con menos ventas.
- Las **personas mayores de 40 años** son las que más compran en estas categorías, mientras que las personas de entre **30 y 33 años** son las que menos compran.
- El **total_amount** (monto total de las transacciones) se encuentra generalmente por debajo de **250**, con algunos outliers cercanos a **2000**.

## 6. **Creación y Evaluación del Modelo Predictivo**

Se probaron varios modelos de Machine Learning y sus resultados fueron los siguientes:

### **1. Regresión Lineal (Linear Regression)**
- **R²**: 0.842
- **RMSE**: 228.45
- **MAE**: 186.05

**Interpretación**: La regresión lineal muestra un **R²** aceptable (0.84), pero los errores absolutos son relativamente altos (MAE de 186). Este modelo puede ser útil si se busca algo sencillo y rápido, pero no es ideal para los datos si se busca mejorar la precisión de las predicciones.

### **2. XGBoost**
- **R²**: 0.9990
- **RMSE**: 18.21
- **MAE**: 13.26

**Interpretación**: XGBoost muestra un **R²** muy alto (0.9990), indicando una excelente capacidad para explicar la variabilidad en los datos. Los valores de RMSE y MAE también son bajos, lo que sugiere que las predicciones son bastante precisas. Este modelo parece ser uno de los más prometedores en términos de rendimiento y generalización.

### **3. Random Forest**
- **R²**: 1.0000
- **RMSE**: 0.0000
- **MAE**: 0.0000

**Interpretación**: Aunque el **R²** es perfecto, con **RMSE** y **MAE** de 0, es posible que el modelo esté **sobreajustado**. Esto significa que probablemente ha "memorizado" los datos de entrenamiento en lugar de generalizar bien a nuevos datos. Este no es un buen resultado en un contexto real, ya que no garantiza que el modelo funcione bien con nuevos datos.

### **4. Regresión Ridge**
- **R²**: 0.842
- **RMSE**: 228.46
- **MAE**: 186.04

**Interpretación**: Al igual que la regresión lineal, la regresión Ridge ofrece un **R²** similar (0.84), pero con errores absolutos relativamente altos. A pesar de que Ridge regulariza el modelo y puede ser útil si tienes multicolinealidad, no es tan preciso como otros modelos como XGBoost.

### **5. Gradient Boosting**
- **R²**: 1.0000
- **RMSE**: 1.69
- **MAE**: 1.03

**Interpretación**: **Gradient Boosting** también tiene un **R²** perfecto (1.0000) y una muy baja RMSE y MAE. A diferencia de Random Forest, no parece estar sobreajustado y sus métricas son más consistentes, lo que sugiere una buena capacidad de generalización.

### **6. Support Vector Regression (SVR)**
- **R²**: 0.9990
- **RMSE**: 17.75
- **MAE**: 5.07

**Interpretación**: **SVR** muestra un **R²** excelente (0.9990), con valores de RMSE y MAE muy bajos. Sin embargo, el modelo puede ser más sensible a la escala de los datos y requiere un ajuste fino de los hiperparámetros para obtener el mejor rendimiento. En este caso, su rendimiento es bastante bueno.

---

## 7. **Predicción de Ventas**

Se entrenó el modelo de **Gradient Boosting** para predecir las ventas futuras en función de los datos históricos almacenados en la tabla Delta. Las predicciones se hicieron tanto a nivel mensual como diario.

## 8. **Despliegue en Streamlit**

El modelo fue desplegado en una aplicación interactiva utilizando **Streamlit**, lo que permitió visualizar las predicciones de ventas de manera dinámica y accesible.

## 9. **Herramientas de Organización Utilizadas**

Durante el desarrollo del proyecto, se utilizaron herramientas de organización como **Notion**, **Miro** , **cronogramas de actividades** para planificar y hacer seguimiento de las tareas.



# Nosotros



<table align='center'>
  <tr>
    <td align='center'>
      <div >
        <a href="https://github.com/JPjuanaponte" target="_blank" rel="author">
          <img width="110" src="https://raw.githubusercontent.com/UrbanGreenSolutions/BlueTrips/main/Assets/IMG-Perfil/01.png"/>
        </a>
        <a href="https://github.com/JPjuanaponte" target="_blank" rel="author">
          <h4 style="margin-top: 1rem;">Juan Aponte </br><small></small></h4>
        </a>
        <div style='display: flex; flex-direction: column'>
        <a href="https://github.com/JPjuanaponte" target="_blank">
          <img style='width:8rem' src="https://img.shields.io/static/v1?style=for-the-badge&message=GitHub&color=172B4D&logo=GitHub&logoColor=FFFFFF&label="/>
        </a>
        <a href="https://www.linkedin.com/in/juan-pablo-aponte-murcia-36603627a/" target="_blank">
          <img style='width:8rem' src="https://img.shields.io/badge/linkedin%20-%230077B5.svg?&style=for-the-badge&logo=linkedin&logoColor=white"/>
        </a>
        </div>
      </div>
    </td>