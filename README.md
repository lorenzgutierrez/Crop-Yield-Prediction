# Modelo de prediccion de produccion de cultivos

El presente cpnsiste en el desarrollo de una prueba tecnica. En la misma se pide la prediccion de la produccion de diferentes cultivos para diversas regiones.

En pos de lograr eso, se implementaron 4 modelos:

1. Random Forest
2. Linear Regressor
3. Lasso
4. Ridge

Random Forest fue el que mejor funcionó, mientras que los otros 3 arrojaron resultados similares. Los detalles de cada modelo pueden verse en la siguiente tabla:

| Model         | Train_R2 | Test_R2  | Test_RMSE     | Test_MAPE   | Bias_pct       |
|---------------|----------|----------|---------------|-------------|----------------|
| Random Forest | 0.996240 | 0.935854 | 22540.222680  | 21.509802   | -6718.53      |
| Linear        | 0.465958 | 0.464365 | 63954.141381  | 166.574758  | -10316.003336  |
| Ridge         | 0.465955 | 0.464359 | 63954.531699  | 166.538231  | -10317.469999  |
| Lasso         | 0.465882 | 0.464345 | 63955.353686  | 166.401606  | -10324.215008  |

El mejor modelo de la corrida del pipeline queda guardado en models, ademas de la metadata correspondiente. Se detecta un ligero overfitting en RF, no afecta drasticamente los resultados. Tambien se aprecia un Bias general, el modelo de RF tiende a subpredecir el valor real, esto se ve claramente en la figura de la prediccion global.

Nota: En Sandbox, se encuentra mayoritariamente mi linea de razonamiento. De ese analisis obtuve que, a nivel global, el comportamiento de la produccion por cultivo es lineal. Puede predecirse con bastante grado de exactitud la produccion con un modelo del estilo $y = m*t + b$. Aqui los resultados por cultivo

| Cultivo              | Slope       | Intercept    | R_value     | P_value     |
|----------------------|-------------|--------------|-------------|-------------|
| Maize                | 6,40316E+15 | -1,24459E+16 | 0.98        | 3,96653E-25 |
| Potatoes             | 1,83668E+15 | -3,50506E+16 | 0.99        | 2,65434E-40 |
| Rice, paddy          | 3,30789E+15 | -6,27672E+15 | 0.98        | 4,16993E-30 |
| Wheat                | 3,42668E+16 | -6,57384E+15 | 0.99        | 9,72455E-33 |
| Sorghum              | 2,18887E+15 | -4,18424E+15 | 0.98        | 1,06478E-25 |
| Soybeans             | 1,45706E+16 | -2,76096E+16 | 0.96        | 1,23352E-15 |
| Cassava              | 5,00641E+15 | -9,02873E+15 | 0.90        | 5,86907E-07 |
| Yams                 | 2,0494E+16  | -3,26837E+15 | 0.75        | 1,50375E-11 |
| Sweet potatoes       | 5,4342E+15  | -9,90918E+15 | 0.94        | 2,03786E-28 |
| Plantains and others | 4,22147E+16 | -7,53564E+15 | 0.78        | 4.93585E-13 |

Esto me resulto particularmente interesante, aunque al final terminé diseñando un modelo mas avanzado para intentar capturar comportamientos regionales. 

## Tareas no cumplidas

1. API Rest: No se hacerlo, podría intentarlo, pero me tomaría mucho mas tiempo. Hacerlo con IA no me parece adecuado. Todas las tareas realizadas con IA las puedo hacer yo, pero me resulta tedioso.
2. Containerizacion: Requiero tiempo
3. Logging de experimentos: Requiero mas tiempo.
4. Data versioning: No se que es esto.

## Disclaimer

### Uso de IA

A la hora de realizar esta prueba utilicé IA. En general hay 4 casos en los que ha de asumirse que usé una IA mayoritariamente para generar codigo:

1. Graficos
2. Logging
3. Unit Tests
4. Docstrings Basicos
5. Config file

Hay pequeños pedazos de codigo que no he generado directamente yo, en estas situaciones queda aclarado en el codigo.

### No uso de IA

En qué casos NO generé codigo con IA (o uso minimo):

1. Desarrollo general de la ingesta de datos
2. Desarrollo general de la limpieza de datos
3. Analisis de los datos
4. Busqueda y generacion de Features
5. Implementacion de los modelos y calculos de metricas
6. Toma de decisiones relevantes para el proyecto (hiperparametros, train test split etc)
7. Generacion del Pipeline.
8. Estructura general del codigo


## Decisiones tecnicas

1. Para hacer la limpieza de datos, busqué no eliminar, en la medida en la que me fue posible, datos de yield_df. Hubo paises y años que tuvieron que ser removidos.
2. Se identificaron 2 años con problemas 1988 y 2003. Para estos casos complete con el promedio de los 5 años anteriores para el campo afectado.
3. Identifique numerosas diferencias de nombres de areas entre los df's. En estos casos usé Fuzzy mathing para completar y recuperar paises. Paises que aun así quedaban incompletos fueron completados con la media.
4. Los datos de yield llegaban hasta 2016, pero rainfall y pesticides hasta 2013. Completé con el promedio en estos casos.
5. Principales features segun mi analisis: Quartiles Pesticides, Rainfall*Temp, Temperatura_optima_cultivo (este ultimo usando IA para estableceer un perfil para cada cultivo)
6. Se hizo Hip Tuning con random forest para disminuir el overfitting existente. Logre mejora, pero sigue existiendo.
7. Use Label Encodign para encodear los features categoricos. Fue mi primera opcion y al ver los resultados del modelo decidi no implementar Target Encoding ni Embeddings. Sin embargo creo que Embeddings podria ayudar a capturar comportamientos entre Areas.
8. Se hace CV para validacion de RF usando como metrica el r-score. Use esta configuracion para intentar disminuir el gap que observe en las primeras iteraciones del modelo. El numero de folds es arbitrario, y mis mejores resultados se obtuvieron con k = 4.
9. Removi los primeros 3 anios del dataset final de entrenamiento. Esto porque use modelos lineales y los features con lags podrian traer problemas.
10. Use las metricas standard: r2, MAE, RMSE, MAPE. Aunque base mis decisiones y analisis principalmente en el r2 score.

## Modo de uso

### Requisitos previos

- Python 3.7 o superior
- pip (gestor de paquetes de Python)

### Instrucciones

1. **Descargar el código**

  ```bash
  git clone [URL_DEL_REPOSITORIO]
  cd [NOMBRE_DEL_DIRECTORIO]

2. **Instalar dependencias**

  ```bash
  pip install -r requirements.txt

3. **Modificar configuracion**

 En config.py, de ser necesario

4. **Ejecucion del pipeline**

 ```bash
   python main.py






