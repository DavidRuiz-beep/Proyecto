# Clasificación de Hogares Pobres en Colombia usando Machine Learning
## Proyecto de Inteligencia Artificial Aplicada

## 1. Planteamiento del Problema

En Colombia, la focalización de subsidios y programas sociales depende de la correcta identificación de los hogares en situación de pobreza. Tradicionalmente, este proceso ha sido realizado mediante metodologías como el SISBÉN, que utiliza variables socioeconómicas para clasificar a la población según su nivel de vulnerabilidad.

Sin embargo, los métodos tradicionales presentan limitaciones asociadas a errores de inclusión (hogares no pobres que reciben subsidios) y errores de exclusión (hogares pobres que quedan por fuera del sistema). Estos problemas generan ineficiencias en la asignación de recursos públicos y afectan la equidad del gasto social.

Ante este contexto, surge la siguiente pregunta de investigación:

¿Puede un modelo de Machine Learning mejorar la precisión en la clasificación de hogares pobres en comparación con métodos tradicionales?

## 2. Objetivo General

Desarrollar y evaluar un modelo de clasificación supervisada que permita identificar hogares en condición de pobreza utilizando variables socioeconómicas, con el fin de mejorar la focalización de subsidios.

## 3. Objetivos Específicos

- Construir una base de datos con variables demográficas y socioeconómicas relevantes.
- Implementar modelos de clasificación como:
  - Regresión Logística
  - K-Nearest Neighbors (KNN)
  - Árboles de Decisión
  - Random Forest
- Comparar el desempeño de los modelos mediante métricas como:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Matriz de Confusión
- Analizar la importancia de las variables predictoras.


## 4. Metodología

El proyecto seguirá las siguientes etapas:

1. Recolección y limpieza de datos.
2. Análisis exploratorio (EDA).
3. Selección de variables predictoras.
4. División del dataset en entrenamiento y prueba.
5. Entrenamiento de modelos de clasificación.
6. Evaluación comparativa de desempeño.
7. Interpretación económica de resultados.


## 5. Dataset

El proyecto podrá utilizar fuentes públicas como:

- Departamento Administrativo Nacional de Estadística (DANE)
- Datos abiertos del Gobierno de Colombia
- Encuesta de Calidad de Vida (ECV)
- Encuesta Integrada de Hogares (GEIH)

Las variables incluirán:

- Ingreso del hogar
- Nivel educativo del jefe de hogar
- Número de personas en el hogar
- Condiciones de vivienda
- Acceso a servicios públicos
- Situación laboral


## 6. Resultados Esperados

Se espera que modelos no lineales como Random Forest o XGBoost presenten mayor capacidad predictiva frente a modelos tradicionales como la Regresión Logística.

El proyecto permitirá:

- Reducir errores de clasificación.
- Mejorar la eficiencia del gasto público.
- Proponer una herramienta basada en datos para política social.


## 7. Tecnologías Utilizadas

- Python
- Scikit-Learn
- Pandas
- NumPy
- Matplotlib


## 10. TXT

- pandas==2.0.3
- numpy==1.24.3
- matplotlib==3.7.1
- seaborn==0.12.2
- openpyxl==3.1.2
- jupyter==1.0.0


## 9. Autor

David Felipe Ruiz Parra
