# Neural Network Sandbox
Un entorno web interactivo desarrollado en JavaScript, HTML5 Canvas y Tailwind CSS, diseñado específicamente para la enseñanza y la experimentación. Este simulador permite abrir la "caja negra" de las Redes Neuronales Artificiales de tipo **Multilayer Perceptron (MLP)**, permitiendo construir, entrenar y diagnosticar modelos en tiempo real sin necesidad de escribir una sola línea de código.

A diferencia de las librerías industriales tradicionales (como TensorFlow o PyTorch), este software prioriza la **comprensión geométrica y conceptual**. El sistema transiciona de forma automática y reactiva entre problemas de **Clasificación Categórica** y **Regresión Numérica Continua**, transformando las matemáticas abstractas en una experiencia visual, orgánica y altamente didáctica.

## Ingesta de Datos y Requisitos del Dataset

El flujo de trabajo comienza en el panel de control izquierdo, donde se realiza la carga del archivo de datos.
* **Formato del Archivo:** El sistema acepta archivos en formato `.csv` planos. Es un requisito mandatorio que la primera fila contenga las cabeceras (*headers*), las cuales la plataforma transformará automáticamente en las claves identificadoras de cada característica (*feature*).
* **Análisis de Tipado Inmediato:** Al cargar el documento, la interfaz procesa el set de datos y muestra un diagnóstico en vivo. Si detecta texto, calcula la cardinalidad de la variable y sus frecuencias relativas; si detecta valores puramente numéricos, extrae los límites físicos absolutos ($x_{min}$ y $x_{max}$) indispensables para los procesos de escalado lineal.

## Configuración Estructural e Ingeniería de Características (Mappers Visuales)

Para cada columna detectada, el usuario dispone de selectores desplegables para definir su rol dinámico en la red (Entrada, Salida, Ignorar) y su método de procesamiento matemático:

### A. Selectores de Normalización Numérica Continua

Para evitar que variables con rangos masivos saturen los gradientes de las neuronas ocultas, se puede asignar visualmente una de dos estrategias de escalado:
* **Norm [0, 1]:** Aplica una compresión lineal acotada al rango de la función Sigmoide mediante la ecuación:
  $$x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$$
* **Norm [-1, 1]:** Centra los datos en cero, un tratamiento ideal si se planea usar la función Tangente Hiperbólica (Tanh) en las capas ocultas para aprovechar su zona de máxima derivada:
  $$x_{norm} = 2 \times \left(\frac{x - x_{min}}{x_{max} - x_{min}}\right) - 1$$

### B. Selectores Cualitativos (Codificación de Categorías)

* **Variables Binarias (Cardinalidad $K = 2$):** Al mapear columnas categóricas de dos estados (ej. `sex` [F/M]), el sistema las comprime automáticamente en **una sola neurona de entrada** discreta $x \in \{0, 1\}$.
* **Variables Multiclase (Cardinalidad $K > 2$):** Al activar el mapeo categórico en columnas con múltiples etiquetas de texto, el simulador ejecuta internamente una transformación *One-Hot Encoding*. Esto expande visualmente la columna en **$K$ neuronas de entrada independientes** representadas por un vector binario esparcido, impidiendo de forma matemática que la red asuma jerarquías o magnitudes numéricas artificiales entre categorías.

## Construcción y Compilación de la Topología

Antes de proceder a la compilación del modelo, el usuario define la partición del dataset mediante el deslizador lineal de porcentaje (ej. 80% Entrenamiento / 20% Prueba). El simulador segmenta las filas de forma aleatoria para evitar sesgos causados por el ordenamiento original del archivo CSV. El set de entrenamiento será utilizado exclusivamente por el motor para calcular los gradientes y actualizar los pesos vectoriales mediante retropropagación, mientras que el set de prueba se mantendrá completamente aislado, actuando como un grupo de control ciego e independiente para evaluar la verdadera capacidad de generalización del grafo al cierre de cada época.

Una vez definido el comportamiento de las columnas al presionar el botón **"Confirmar y Construir Red"**, el sistema compila la arquitectura:

**Dimensión del Vector de Entrada:** El sistema calcula de forma exacta la cantidad de neuronas físicas de la primera capa ($D_{in}$) aplicando la ecuación:
  $$D_{in} = \sum C_{continuas} + \sum B_{binarias} + \sum K_{categoricas}$$

**Dimensión del Vector de Salida (Capa de Output):** De forma análoga, el framework calcula la dimensión final de la capa de salida ($D_{out}$) adaptando la topología según el paradigma del problema definido por las columnas configuradas como `salida`:

* **Regresión Continua o Clasificación Binaria:** Si la columna objetivo es numérica o categórica de dos estados ($K=2$), se asigna exactamente una neurona por cada variable independiente mapeada.

* **Clasificación Multiclase:** Si la columna objetivo es categórica con una cardinalidad $M > 2$, el sistema expande el espacio dimensional a $M$ neuronas de salida, preparando la capa para la distribución de probabilidad Softmax.

* **Ecuación general para la geometría de salida:**
  $$D_{out} = \sum M_{categóricas} + \sum 1_{continuas / binarias}$$

## Configuración Global de Hiperparámetros

Una vez estructurada la geometría de entrada y salida, el usuario dispone en la sección central del panel de un conjunto de controles deslizantes (sliders) y menús desplegables para manipular los hiperparámetros del algoritmo de optimización antes y durante la ejecución física de la red:

* **Tasa de Aprendizaje (LR):** Regulada mediante un control deslizante continuo con un rango de 0.001 a 0.5 y saltos discretos de 0.001. Define la magnitud del paso escalar ($\eta$) que darán los pesos vectoriales en la dirección opuesta al gradiente en cada iteración. Deslizar el control hacia valores altos acelera los cambios iniciales pero puede inducir divergencia matemática; desplazarlo hacia el mínimo de 0.001 garantiza estabilidad a costa de una convergencia mucho más lenta.

* **Momento (Inercia):** Ajustable a través de un slider que abarca desde 0.0 hasta 0.99 (pasos de 0.01). Permite al usuario configurar la inercia plástica de la optimización. Al incrementar este valor, el sistema retiene una fracción ($\alpha$) del vector de actualización anterior ($v_{t-1}$), permitiendo que los pesos "deslicen" con velocidad a través de zonas llanas de la superficie de error o salten valles locales poco profundos:

  $$v_{t} = \alpha v_{t-1} + \eta \frac{\partial E}{\partial w}$$
  $$w_{t} = w_{t-1} - v_{t}$$

* **Tasa de Dropout:** Controlado por un selector deslizante de 0.0 a 0.6 con incrementos de 0.05. Determina la probabilidad estocástica ($p$) de apagar temporalmente neuronas en las capas ocultas durante cada minilote. Configurar este valor por encima de 0.0 obliga a la red a no depender de rutas sinápticas específicas, destruyendo la co-adaptación y forzando al grafo a co-entrenar caminos redundantes y robustos.

* **Regularización L2 (Decay):** Modificable mediante un control de alta precisión que oscila entre 0.0 y 0.01 con pasos milimétricos de 0.0001. Aplica una penalización cuadrática constante a la magnitud de los pesos en cada actualización. Elevar este parámetro actúa como una fuerza de fricción matemática que atrae los valores sinápticos hacia el cero, impidiendo que aristas individuales tomen valores desproporcionados, limitando así la memorización del ruido del set de entrenamiento.

* **Tamaño del Lote (Batch Size):** Un menú desplegable que define la cantidad de muestras del CSV que el motor procesará en paralelo antes de calcular el gradiente promedio y aplicar una actualización física en los parámetros de la red:

  * **1 (Estocástico):** Actualiza los pesos fila por fila. Introduce una alta volatilidad matemática que ayuda a escapar de mínimos locales, pero genera fluctuaciones drásticas en las gráficas de control.
  * **16 (Mini-lote) / 32 (Estándar) / 64 (Lote Grande):** Opciones híbridas estándar. Balancean la estabilidad del descenso del gradiente con la eficiencia computacional del sistema.
  * **Full Batch (Todo el CSV):** Identificado con el valor -1. Acumula los gradientes de la totalidad del conjunto de entrenamiento antes de ejecutar una sola modificación en los pesos, garantizando un descenso determinista y suave sobre la superficie de costo a expensas de un mayor costo computacional por época.

* **Función de Pérdida (Loss):** Menú de selección interactiva que define el criterio matemático de coste para el motor. El usuario debe emparejar este selector con la naturaleza de la capa de salida: Error Cuadrático Medio (MSE) para aproximaciones numéricas continuas (regresión), Entropía Cruzada Categórica (CCE) para clasificaciones probabilísticas excluyentes o Entropía Cruzada Binaria (BCE) para problemas de clasificación binaria.

* **Inicialización de Pesos:** Selector desplegable que permite elegir el algoritmo de distribución estadística encargado de romper la simetría inicial de las matrices de peso antes de presionar el botón de inicio:

  * **Aleatoria Uniforme:** Inyecta valores puramente al azar distribuidos uniformemente entre -1 y 1. Es la técnica tradicional, pero altamente propensa a saturar o apagar neuronas en arquitecturas densas o profundas.
  * **Xavier / Glorot (Uniforme):** Escala los límites de la distribución uniforme basándose en la inversa de la raíz cuadrada de la suma de las neuronas de entrada y salida de la capa. Es la opción óptima y recomendada cuando se trabaja con funciones de activación Sigmoide o Tanh en las capas ocultas.
  * **He Initialization (Uniforme):** Modifica el escalado de los pesos considerando exclusivamente la cantidad de neuronas de la capa anterior. Diseñada específicamente para evitar el fenómeno del desvanecimiento del gradiente cuando se implementa la función de activación ReLU.

## Métricas del modelo
La interfaz muestra en tiempo real un conjunto de métricas que permiten al usuario evaluar la evolución del entrenamiento y la capacidad de generalización del modelo:
* **Pérdida de Entrenamiento (Loss):** Calculada al final de cada época, representa el valor promedio de la función de pérdida sobre el conjunto de entrenamiento. Una disminución constante indica que la red está aprendiendo correctamente.
* **Precisión Total:** Para problemas de clasificación, esta métrica indica el porcentaje de predicciones correctas sobre el total de muestras evaluadas (Accuracy), se calcula tanto para el conjunto de entrenamiento como para el de prueba. Para problemas de regresión, se muestra el error promedio (MAE).

## Inspector de componentes

El panel derecho de la interfaz contiene un inspector visual que permite al usuario examinar en detalle los componentes internos de la red neuronal, mostrando el tipo de elemento y su identificador único. Al hacer clic en cualquier neurona, conexión o capa, se despliega un cuadro de información que incluye:

### Capas
* **Función de Activación:** Indica la función matemática utilizada en la capa, al cambiar desde esta opción el método de activación, el sistema coloca automáticamente esta función en todas las neuronas de la capa seleccionada. Las opciones disponibles son:
  * **Ninguna (identidad):** La salida es igual a la entrada, sin transformación.
  * **ReLu:** La salida es cero para entradas negativas y lineal para entradas positivas, útil para evitar el desvanecimiento del gradiente.
  * **Sigmoide:** La salida está acotada entre 0 y 1, útil para problemas de clasificación binaria.
  * **Tangente Hiperbólica (Tanh):** La salida está acotada entre -1 y 1, útil para problemas de regresión y clasificación multiclase.
  * **Softmax:** La salida es un vector de probabilidades que suman 1, útil para problemas de clasificación multiclase.

* **Añadir neuronas:** Permite al usuario modificar la cantidad de neuronas en la capa seleccionada, ajustando la capacidad de representación del modelo.
* **Agregar capas a la izquierda o derecha:** Permite al usuario insertar nuevas capas en la arquitectura de la red, aumentando su profundidad y complejidad.
* **Eliminar capa:** Permite al usuario eliminar la capa seleccionada, reduciendo la complejidad del modelo.

### Neuronas
* **Valor de activación:** Muestra el valor de activación actual de la neurona.
* **Entrada neta:** Muestra la suma ponderada de las entradas antes de aplicar la función de activación.
* **Sesgo (bias):** Muestra y permite modificar el valor del sesgo asociado a la neurona.
* **Función de activación:** Permite cambiar la función de activación de la neurona individualmente, a exepción de que la capa tenga una función de activación de capa completa como Softmax.
* **Eliminar neurona:** Permite al usuario eliminar la neurona seleccionada, ajustando la capacidad de representación del modelo.

### Conexiones
* **Valor de activación:** Muestra el valor de activación actual de la conexión.
* **Peso (weight):** Muestra y permite modificar el valor del peso asociado a la conexión.
* ***Eliminar conexión:** Permite al usuario eliminar la conexión seleccionada, ajustando la arquitectura de la red.

## Entrenamiento

El control del bucle de entrenamiento se gestiona de forma interactiva mediante los botones del panel central.

* **Paso a Paso:** Permite al usuario ejecutar una sola iteración del algoritmo de retropropagación, actualizando los pesos y sesgos de la red. Esto es útil para observar cómo cambian los valores internos de la red en respuesta a un solo lote de datos.
* **Entrenamiento Continuo:** Permite al usuario ejecutar el entrenamiento de la red de forma continua, actualizando los pesos y sesgos de la red en cada iteración hasta que se detenga manualmente. Esto es útil para observar cómo la red converge hacia una solución óptima a lo largo del tiempo.
* **Detener Entrenamiento:** Permite al usuario detener el entrenamiento continuo de la red en cualquier momento, congelando los valores actuales de los pesos y sesgos. Esto es útil para pausar el entrenamiento y analizar los resultados obtenidos hasta ese punto.
* **Reiniciar Entrenamiento:** Permite al usuario reiniciar el entrenamiento de la red desde cero, restableciendo los pesos y sesgos a sus valores iniciales. Esto es útil para comparar diferentes configuraciones de hiperparámetros o arquitecturas de red.

## Probar el modelo
Mediante el botón **"Probar Modelo"**, el usuario puede evaluar la capacidad de generalización del modelo entrenado utilizando el conjunto de prueba previamente definido. Al presionar este botón se muestra una ventana emergente con los campos de entrada del modelo, además de la opción de aplicar automaticamente a los datos la normalización definida originalmente, permitiendo al usuario ingresar nuevos datos y obtener predicciones en tiempo real. Esto permite verificar cómo el modelo se comporta con datos que no ha visto durante el entrenamiento, proporcionando una medida de su capacidad de generalización.

## Guardar y Cargar Modelos
El botón **"Guardar Modelo"** permite al usuario exportar la arquitectura y los pesos entrenados, las configuraciones y datos de entrenamiento y prueba a un archivo JSON. El botón **"Cargar Modelo"** permite al usuario importar un archivo JSON previamente guardado, restaurando la arquitectura, los pesos y las configuraciones del modelo para continuar el entrenamiento o realizar nuevas pruebas.