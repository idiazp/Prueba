# **Aprendizaje supervisado**
# SL03. Regresión Lineal

## <font color='blue'>**Regresion lineal simple**</font>

Comenzaremos con la regresión lineal más familiar, un ajuste de línea recta a los datos.
Un ajuste en línea recta es un modelo de la forma
$$
y = ax + b
$$
donde $a$ se conoce comúnmente como *pendiente*, y $b$ se conoce comúnmente como *intersección*.

Considere los siguientes datos, que se encuentran dispersos sobre una línea con una pendiente de 2 y una intersección de -5:

## Metodos de los minimos cuadrados

1. Planteamos el modelo lineal:
\begin{equation}
y = ax + b
\end{equation}

2. Definimos la función de costo que mide la suma de las diferencias al cuadrado entre los valores observados $(y_i$) y los valores predichos ($\hat{y}_i$), dados por $\hat{y}_i = ax_i + b$. La función de costo es:

\begin{equation}
J(a, b) = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \sum_{i=1}^{n} (y_i - (ax_i + b))^2
\end{equation}

3. Derivamos parcialmente $J(a, b)$ con respecto a $a$ y $b$ y establecemos las derivadas iguales a cero. Derivando con respecto a $a$ y estableciendo la derivada igual a cero:

\begin{equation}
\frac{\partial J}{\partial a} = -2 \sum_{i=1}^{n} x_i(y_i - (ax_i + b)) = 0
\end{equation}

Resolviendo esta ecuación, obtenemos:

\begin{equation}
\sum_{i=1}^{n} x_iy_i = a \sum_{i=1}^{n} x_i^2 + b \sum_{i=1}^{n} x_i
\end{equation}

Derivando con respecto a $b$ y estableciendo la derivada igual a cero:
\begin{equation}
\frac{\partial J}{\partial b} = -2 \sum_{i=1}^{n} (y_i - (ax_i + b)) = 0
\end{equation}
Resolviendo esta ecuación, obtenemos:
\begin{equation}
\sum_{i=1}^{n} y_i = a \sum_{i=1}^{n} x_i + n b
\end{equation}

4. Resolvemos este sistema de ecuaciones lineales para obtener las fórmulas para $a$ y $b$:

\begin{equation}
a = \frac{n \sum_{i=1}^{n} x_iy_i - \sum_{i=1}^{n} x_i \sum_{i=1}^{n} y_i}{n \sum_{i=1}^{n} x_i^2 - (\sum_{i=1}^{n} x_i)^2}
\end{equation}


//

\begin{equation}
b = \frac{\sum_{i=1}^{n} y_i \sum_{i=1}^{n} x_i^2 - \sum_{i=1}^{n} x_i \sum_{i=1}^{n} x_iy_i}{n \sum_{i=1}^{n} x_i^2 - (\sum_{i=1}^{n} x_i)^2}
\end{equation}


### **Implementacion en Python: El metodo de los minimos cuadrados**

## <font color='green'>Actividad 1</font>

Ejercicio: Regresión Polinómica y Validación Cruzada

Dado un conjunto de datos sintéticos con ruido, implemente un modelo de regresión polinómica utilizando el método de los mínimos cuadrados y valida el modelo utilizando validación cruzada.

 <font color='green'>Fin Actividad 1</font>

grado = 2
Cross validation scores: [0.98921704 0.97204274 0.98896651 0.98321935 0.97327693]
Mean cross validation score: 0.9813445163269325

<font color='green'>Fin Actividad 1</font>

## Implementando con sklearn

Usaremos el estimador de Scikit-Learn ``LinearRegression`` para ajustar la data y construir el modelo:

La pendiente y la intersección de los datos están contenidos en los parámetros de ajuste del modelo, que en Scikit-Learn siempre están marcados con un guión bajo al final.
Aquí los parámetros relevantes son `` coef_`` e `` intercept_``:

Vemos que los resultados están muy cerca de las entradas, como podríamos esperar.

Sin embargo, el estimador de ``LinearRegression`` es mucho más capaz que esto; además de ajustes simples en línea recta, también puede manejar modelos lineales multidimensionales de la forma
$$
y = a_0 + a_1 x_1 + a_2 x_2 + \cdots
$$
donde hay multiples valores $x$.
Geométricamente, esto es similar a ajustar un plano a puntos en tres dimensiones, o ajustar un hiperplano a puntos en dimensiones más altas.

La naturaleza multidimensional de tales regresiones las hace más difíciles de visualizar, pero podemos ver uno de estos ajustes en acción construyendo algunos datos de ejemplo, usando el operador de multiplicación de matrices de NumPy:

Aquí, los datos $ y $ se construyen a partir de tres valores $ x $ aleatorios, y la regresión lineal recupera los coeficientes utilizados para construir los datos.

De esta manera, podemos usar el estimador único de ``LinearRegression``  para ajustar líneas, planos o hiperplanos a nuestros datos.
Todavía parece que este enfoque se limitaría a relaciones estrictamente lineales entre variables, pero resulta que también podemos relajar esto.

## <font color='green'>Actividad 2</font>

1. Realice un 5-fold cross validation con el modelo de regresión lineal con los datos anteriores (*X* e *y*).
2. Evalúe el error para los distintos conjuntos de test.

¿Qué tan estable es el modelo?, ¿Cuán bueno es el error?


<font color='green'>Fin Actividad 1</font>

## <font color='blue'>**Regresión de funciones base**</font>

Sea $H$ una familia de funciones $X \rightarrow Y$ y $T = \{(x_1,y_1),.....(x_n,y_n)\}$  un training set. Donde $x_i \in \mathbb{R}^D$ y $y_i \in \mathbb{R}$. Entonces se quiere seleccionar la función $g \in H$ que minimiza el error dado por:

$$ E(g) = \frac{1}{2} \sum_{i=1}^n (g(x_i) - y_i)^2$$

En el caso de las funciones lineales nuestro H corresponde a:

$$ \{h(x): h(x) = w_o + \sum_{i=1}^D w_ix_i \}$$

¿Cómo podemos expandir nuestro H pero que el mecanismo lineal siga
funcionando?

Sea:
 $$\phi_1,....,\phi_M:\mathbb{R}^D \rightarrow \mathbb{R}$$

Entonces un elemento $h \in H$ tiene la forma de:
$$ h(x) = w_o + \sum_{i=1}^M w_i\phi_i(x)$$

Al conjunto de funciones $\phi_i$, lo llamaremos funciones base.




Un truco que puede utilizar para adaptar la regresión lineal a relaciones no lineales entre variables es transformar los datos de acuerdo con *funciones base*.

La idea es tomar nuestro modelo lineal multidimensional:
$$
y = a_0 + a_1 x_1 + a_2 x_2 + a_3 x_3 + \cdots
$$
y construir $x_1, x_2, x_3,$, de nuestra entrada unidimiensional $x$.
Esto es, $x_n = f_n(x)$, donde $f_n()$ es una funcion que transforma la data.

Por ejemplo, si $f_n(x) = x^n$, nuestro modelo se transforma en una regresion polinomial:
$$
y = a_0 + a_1 x + a_2 x^2 + a_3 x^3 + \cdots
$$
Observe que este es *todavía un modelo lineal*: la linealidad se refiere al hecho de que los coeficientes $a_n$ nunca se multiplican ni se dividen entre sí.
Lo que hemos hecho efectivamente es tomar nuestros valores unidimensionales $x$ y proyectarlos en una dimensión superior, de modo que un ajuste lineal pueda encajar relaciones más complicadas entre $x$ y $y$.

### Funciones de base polinomial

Esta proyección polinomial es lo suficientemente útil como para estar integrada en Scikit-Learn, utilizando el transformador ``PolynomialFeatures``:

Vemos aquí que el transformador ha convertido nuestra matriz unidimensional en una matriz tridimensional tomando el exponente de cada valor.
Esta nueva representación de datos de mayor dimensión se puede conectar a una regresión lineal.


Con esta transformación en su lugar, podemos usar el modelo lineal para ajustar relaciones mucho más complicadas entre $x$ y $y$.
Por ejemplo, aquí hay una onda sinusoidal con ruido:

Nuestro modelo lineal, mediante el uso de funciones de base polinomial de séptimo orden, puede proporcionar un ajuste excelente a estos datos no lineales.

### Funciones base gaussianas
Por supuesto, son posibles otras funciones básicas.
Por ejemplo, un patrón útil es ajustar un modelo que no es una suma de bases polinomiales, sino una suma de bases gaussianas.
El resultado podría parecerse a la siguiente figura:

![](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/figures/05.06-gaussian-basis.png?raw=1)
[figure source in Appendix](#Gaussian-Basis)

Las regiones sombreadas en el gráfico son las funciones base escaladas y, cuando se suman, reproducen la curva suave a través de los datos.
Estas funciones de base gaussiana no están integradas en Scikit-Learn, pero podemos escribir un transformador personalizado que las creará, como se muestra aquí y se ilustra en la siguiente figura (los transformadores de Scikit-Learn se implementan como clases de Python; leer la fuente de Scikit-Learn es una buena forma de ver cómo se pueden crear):

## <font color='green'>Actividad 3</font>

Entienda el código y describa la utilidad de cada uno de los metodos definidos.

Ayuda:
1. Scikit-Learn nos proporciona dos excelentes clases base, TransformerMixin y BaseEstimator. Heredar de TransformerMixin asegura que todo lo que tenemos que hacer es escribir nuestros métodos de fit  y transform y obtenemos fit_transform de forma gratuita. La herencia de BaseEstimator garantiza que obtengamos get_params y set_params de forma gratuita. Dado que el método fit  no necesita hacer nada más que devolver el objeto en sí, todo lo que realmente necesitamos hacer después de heredar de estas clases es definir el método de transformación para nuestro transformador personalizado y obtenemos un transformador personalizado completamente funcional que puede ser sin problemas integrado con una canalización de scikit-learn! Fácil.

2. Las bases gaussianas tienen la siguiente forma. $\phi_j = wj*exp(-\frac {(x - \mu_j)^2}{2\sigma^2})$

3. Puede mirar el siguiente link. https://towardsdatascience.com/custom-transformers-and-ml-data-pipelines-with-python-20ea2a7adb65


<font color='green'>Fin Actividad 3</font>

Ponemos este ejemplo aquí solo para aclarar que no hay nada mágico en las funciones de base polinómica: si tiene algún tipo de intuición en el proceso de generación de sus datos que le hace pensar que una base u otra podría ser apropiada, puede usarlas como bien.

## <font color='blue'>**Regularization**</font>

La introducción de funciones base en nuestra regresión lineal hace que el modelo sea mucho más flexible, pero también puede conducir muy rápidamente a un ajuste excesivo.
Por ejemplo, si elegimos demasiadas funciones de base gaussiana, terminamos con resultados que no se ven tan bien:

Con los datos proyectados a la base de 30 dimensiones, el modelo tiene demasiada flexibilidad y llega a valores extremos entre ubicaciones donde está limitado por los datos.
Podemos ver la razón de esto si graficamos los coeficientes de las bases gaussianas con respecto a sus ubicaciones:

El panel inferior de esta figura muestra la amplitud de la función base en cada ubicación.
Este es un comportamiento de sobreajuste típico cuando las funciones base se superponen: los coeficientes de las funciones base adyacentes explotan y se cancelan entre sí.
Sabemos que tal comportamiento es problemático, y sería bueno si pudiéramos limitar explícitamente tales picos en el modelo penalizando los valores grandes de los parámetros del modelo.
Esta penalización se conoce como *regularización* y se presenta en varias formas.

### Ridge regression ($L_2$ Regularization)

Quizás la forma más común de regularización se conoce como *ridge regression* o $ L_2 $ *regularización*, a veces también llamada *regularización de Tikhonov*.
Esto procede penalizando la suma de cuadrados (norma L2) de los coeficientes del modelo; en este caso, la penalización en el ajuste del modelo sería
$$
P = \alpha\sum_{n=1}^N \theta_n^2
$$
donde $\alpha$ es un parametro libre que controla la fuerza de la penalidad.


El parámetro $\alpha$ es esencialmente una perilla que controla la complejidad del modelo resultante.
En el límite $\alpha\ \to 0$, recuperamos el resultado de regresión lineal estándar; en el límite $\alpha \to \infty$, se suprimirán todas las respuestas del modelo.
Una ventaja de la regresión de crestas en particular es que se puede calcular de manera muy eficiente, a un costo computacional apenas mayor que el modelo de regresión lineal original.

### Lasso regression ($L_1$ regularization)

Otro tipo de regularización muy común se conoce como lazo, e implica penalizar la suma de valores absolutos (1-norma) de los coeficientes de regresión:
$$
P = \alpha\sum_{n=1}^N |\theta_n|
$$
Aunque esto es conceptualmente muy similar a la regresión de crestas, los resultados pueden diferir sorprendentemente: por ejemplo, debido a razones geométricas, la regresión de lazo tiende a favorecer *modelos dispersos* cuando es posible: es decir, preferentemente establece los coeficientes del modelo exactamente a cero.

Podemos ver este comportamiento al duplicar la figura de regresión de la cresta, pero usando coeficientes normalizados L1:

Con la penalización por regresión de lazo, la mayoría de los coeficientes son exactamente cero, y el comportamiento funcional está modelado por un pequeño subconjunto de las funciones básicas disponibles.
Al igual que con la regularización de crestas, el parámetro $\alpha$ ajusta la fuerza de la penalización y debe determinarse mediante, por ejemplo, validación cruzada.

## <font color='green'>Actividad 4</font>

Usaremos el *Combined Cycle Power Plant Data Set*.

1. Implementa distintos modelos de regresión para predecir el Energy Output (EP) de la planta y compare sus resultados. Los parametros entregados son los siguientes:
  * Ambient Temperature (AT)
  * Exhaust Vaucum (V)
  * Ambient Pressure (AP)
  * Relative Humidity (RH)

2. Grafique sus resultados.
3. Calcular el coeficiente de correlación de Pearson

<font color='green'>Fin Actividad 3</font>

## **Regresion Lineal: EL metodo gradiente descendiente**

1. Se define una función de costo, que en el caso de la regresión lineal es el error cuadrático medio (MSE). Si tenemos un conjunto de datos con n observaciones $(x_1, y_1), \ldots, (x_n, y_n)$ y queremos ajustar una recta $y = ax + b$, la función de costo es:

\begin{equation}
J(a,b) = \frac{1}{n} \sum_{i=1}^{n}(y_i - (ax_i + b))^2
\end{equation}

2. Luego se calcula el gradiente de la función de costo con respecto a los parámetros a y b. Las derivadas parciales son:

\begin{equation}
\frac{\partial J}{\partial a} = -\frac{2}{n} \sum_{i=1}^{n}x_i(y_i - (ax_i + b))
\end{equation}

\begin{equation}
\frac{\partial J}{\partial b} = -\frac{2}{n} \sum_{i=1}^{n}(y_i - (ax_i + b))
\end{equation}


3. $$(\frac{\partial J}{\partial a}, \frac{\partial J}{\partial b})$$

3. A continuación, actualizamos los parámetros a y b en la dirección opuesta del gradiente. Para ello, utilizamos un parámetro de aprendizaje $\alpha$:

\begin{equation}
a_{\text{new}} = a_{\text{old}} - \alpha \frac{\partial J}{\partial a}
\end{equation}

\begin{equation}
b_{\text{new}} = b_{\text{old}} - \alpha \frac{\partial J}{\partial b}
\end{equation}

Repetimos los pasos 2 y 3 hasta que la función de costo converja a un mínimo (que será un mínimo local o global) o hasta que se alcance un número máximo de iteraciones.

Finalmente, los parámetros a y b encontrados son los que minimizan la función de costo, y representan la pendiente y el intercepto de la recta que mejor se ajusta a los datos según el criterio de mínimos cuadrados.

### Partimos con mediciones de (x,y)

$X_b.dot(theta)$

$(1, x_i)(b_i,a_i) = 1*b_i + a_i*x_i$


\begin{equation}
\frac{\partial J}{\partial a} = -\frac{2}{n} \sum_{i=1}^{n}x_i(y_i - (ax_i + b))
\end{equation}

\begin{equation}
\frac{\partial J}{\partial b} = -\frac{2}{n} \sum_{i=1}^{n}(y_i - (ax_i + b))
\end{equation}

## Minimos cuadrados

## <font color='green'>Actividad 5</font>

La regresión lineal se estudió a través del método de los minimos cuadrados. En esta actividad le proponemos el desafío de implementar la regresión lineal a traves del algoritmo de gradiente descendente.

Para esto uste debe aprender cómo funciona  el algoritmo de descenso de gradientes e implementarlo desde cero en Python y aplicarlo al problema de regresión lineal.



<font color='green'>Fin Actividad 5</font>