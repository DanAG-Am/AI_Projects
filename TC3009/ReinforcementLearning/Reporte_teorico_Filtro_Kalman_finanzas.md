# Filtro de Kalman en el sector financiero

## Portada

**Institución:** [Nombre de la institución]  
**Asignatura:** [Nombre de la asignatura]  
**Título:** Análisis teórico del uso del filtro de Kalman en el sector financiero  
**Docente:** [Nombre del docente]  
**Integrantes:** [Nombres y matrículas]  
**Grupo:** [Grupo]  
**Fecha:** [Fecha de entrega]

---

## Abstract

The Kalman filter is a recursive estimation method that combines a dynamic model with noisy observations to infer states that are not directly observable. In finance, it can represent time-varying parameters, such as an asset's sensitivity to the market, and update their estimates as new information becomes available. This report explains the state-space formulation and the filter's prediction and correction steps, then analyzes potential applications to dynamic beta estimation, trend tracking, and changing relationships between assets. It examines Faff, Hillier, and Hillier's (2000) comparison of methods for modeling time-varying conditional betas using daily returns from UK industries. Their results favored a Kalman filter with a random-walk parameterization under the mean-square-error criterion, but do not establish that the method is superior in every market or period. The report also discusses the filter's assumptions, limitations during abrupt changes, and implementation risks. This is a theoretical literature-based review; it presents no original implementation or experimental results.

**Resumen.** El filtro de Kalman es un método recursivo de estimación que combina un modelo dinámico con observaciones ruidosas para inferir estados que no se observan directamente. En finanzas, permite representar parámetros que cambian con el tiempo, como la sensibilidad de un activo al mercado, y actualizar sus estimaciones conforme llega nueva información. Este reporte explica el modelo de espacio de estados y las etapas de predicción y corrección, y analiza posibles usos en la estimación de betas dinámicas, el seguimiento de tendencias y la actualización de relaciones entre activos. Se examina el estudio de Faff, Hillier y Hillier (2000), quienes compararon métodos para modelar betas condicionales variables con retornos diarios de industrias del Reino Unido. Sus resultados favorecieron al filtro de Kalman con una parametrización de caminata aleatoria bajo el criterio de error cuadrático medio, sin demostrar superioridad universal. También se discuten supuestos, limitaciones ante cambios abruptos y riesgos de implementación. El trabajo es una revisión teórica y bibliográfica, sin implementación ni resultados experimentales propios.

**Keywords:** Kalman filter, finance, state-space model, time-varying beta, recursive estimation.  
**Palabras clave:** filtro de Kalman, finanzas, espacio de estados, beta variable en el tiempo, estimación recursiva.

## Introducción

### Definición del problema

En los mercados financieros, las relaciones entre activos, índices y factores de riesgo pueden cambiar con el tiempo. Una beta calculada una sola vez mediante una regresión histórica supone que la sensibilidad de un activo al mercado permanece constante durante el periodo analizado. Esa hipótesis puede ser poco adecuada cuando cambian las condiciones económicas, la composición de una empresa o la percepción del riesgo. Al mismo tiempo, los retornos observados contienen ruido y no revelan directamente el parámetro subyacente que se desea estimar.

El problema que aborda este reporte es explicar cómo puede emplearse el filtro de Kalman para estimar recursivamente estados financieros no observables que evolucionan en el tiempo, y valorar en qué condiciones ese uso resulta útil. El foco principal es la estimación de betas dinámicas, porque existe literatura que compara este enfoque con alternativas econométricas en datos de mercado (Faff et al., 2000).

### Objetivos

**Objetivo general.** Analizar teóricamente el uso del filtro de Kalman en el sector financiero, sus fundamentos, aplicaciones, ventajas y limitaciones.

**Objetivos específicos.**

- Describir la formulación del filtro mediante un modelo de espacio de estados.
- Explicar la estimación de una beta financiera variable en el tiempo.
- Revisar aplicaciones financieras plausibles y evidencia publicada relevante.
- Discutir los supuestos, riesgos y criterios que deberían considerarse al evaluar el método.

### Alcance y división del reporte

El reporte es una revisión explicativa, no un estudio empírico. No se descargaron series financieras, no se estimó un modelo propio ni se generaron resultados. Primero se presenta el fundamento matemático; después se analiza su uso financiero, con énfasis en betas dinámicas; luego se discuten la interpretación y las limitaciones, y finalmente se presentan conclusiones, referencias y un anexo para completar la división real del trabajo.

## Análisis del uso del filtro de Kalman en el sector financiero

### Fundamento: estados no observables y observaciones ruidosas

El filtro de Kalman, introducido por Kalman (1960), resuelve un problema de estimación recursiva para sistemas lineales con ruido, bajo supuestos probabilísticos que habitualmente incluyen errores gaussianos de media cero. Distingue entre el **estado** que se quiere conocer y la **observación** disponible. En una aplicación financiera, el estado puede ser una beta, un nivel de tendencia o un coeficiente de cobertura, mientras que la observación puede ser un retorno o una diferencia de precios.

Una representación lineal general de espacio de estados es:

$$
y_t = Z_t \alpha_t + d_t + \varepsilon_t,
$$

$$
\alpha_t = T_t \alpha_{t-1} + c_t + \eta_t.
$$

Aquí, $y_t$ es el vector de observaciones; $\alpha_t$ es el vector de estados latentes; $Z_t$ relaciona el estado con la observación; $T_t$ describe la evolución del estado; $d_t$ y $c_t$ son términos determinísticos; y $\varepsilon_t$ y $\eta_t$ son, respectivamente, el ruido de observación y el ruido del proceso. Sus covarianzas se representan normalmente mediante $R_t$ y $Q_t$. La matriz $Q_t$ controla cuánto pueden cambiar los estados entre periodos; $R_t$ representa la incertidumbre atribuida a la medición.

El algoritmo alterna dos etapas:

1. **Predicción.** A partir del estado estimado en $t-1$, proyecta el estado y su incertidumbre hacia $t$:

$$
\hat{\alpha}_{t|t-1}=T_t\hat{\alpha}_{t-1|t-1}+c_t,
$$

$$
P_{t|t-1}=T_tP_{t-1|t-1}T_t^\mathsf{T}+Q_t.
$$

2. **Corrección.** Cuando llega la observación $y_t$, compara el dato con la predicción y actualiza la estimación:

$$
K_t=P_{t|t-1}Z_t^\mathsf{T}(Z_tP_{t|t-1}Z_t^\mathsf{T}+R_t)^{-1},
$$

$$
\hat{\alpha}_{t|t}=\hat{\alpha}_{t|t-1}+K_t(y_t-Z_t\hat{\alpha}_{t|t-1}-d_t).
$$

$P$ representa la covarianza del error de estimación y $K_t$ es la ganancia de Kalman: pondera la predicción y la información nueva de acuerdo con sus incertidumbres relativas. Una observación muy ruidosa suele recibir menor peso; una estimación previa incierta permite que la observación nueva tenga mayor influencia. Esta actualización secuencial evita recalcular desde cero toda la historia en cada periodo, aunque los resultados dependen de la especificación del modelo y de sus parámetros (Durbin & Koopman, 2012; Harvey, 1989).

### Aplicación central: estimación de beta dinámica

En el modelo de valoración de activos CAPM, la beta expresa la sensibilidad del retorno de un activo frente al retorno del mercado. Una versión de beta constante puede escribirse como una regresión lineal. Para permitir que la sensibilidad cambie, puede formularse un modelo de regresión con parámetros variables:

$$
r_{i,t}-r_{f,t}=\alpha_t+\beta_t(r_{m,t}-r_{f,t})+\varepsilon_t,
$$

$$
\begin{bmatrix}\alpha_t\\\beta_t\end{bmatrix}=
\begin{bmatrix}\alpha_{t-1}\\\beta_{t-1}\end{bmatrix}+\eta_t.
$$

$r_{i,t}$ es el retorno del activo, $r_{m,t}$ el retorno del mercado y $r_{f,t}$ la tasa libre de riesgo. La primera ecuación vincula el retorno excedente observado con el estado latente $[\alpha_t,\beta_t]^\mathsf{T}$. La segunda supone una caminata aleatoria: los parámetros de hoy parten de los de ayer y pueden desplazarse por una innovación. Esta es una opción de modelado, no una ley del mercado. Si $Q$ es cercano a cero, la beta se mueve poco y el modelo se aproxima a uno de parámetro estable; un $Q$ mayor permite más adaptación, pero también puede hacer que la estimación reaccione al ruido.

La beta dinámica puede ser útil para análisis de riesgo y asignación de activos: ofrece una estimación que se actualiza al incorporar retornos nuevos y hace visible la evolución temporal de la exposición sistemática. En principio, esa trayectoria puede informar decisiones de cobertura, límites de riesgo o comparación entre periodos. No obstante, una beta estimada es condicional al modelo, al índice elegido como mercado, a la frecuencia de muestreo y al periodo de datos; no es una medida universal ni una garantía sobre retornos futuros.

Faff et al. (2000) estudiaron tres enfoques para modelar betas condicionales variables: modelos GARCH, el modelo de mercado extendido de Schwert y Seguin, y el algoritmo de Kalman. Usando retornos diarios de industrias del Reino Unido, reportaron que el modelo de mercado simple era tan eficiente como los enfoques GARCH más complejos, mientras que el filtro de Kalman con una parametrización de caminata aleatoria dominaba a los demás modelos según el criterio de error cuadrático medio. Los autores también señalaron que combinar métodos podía producir estimadores más potentes de la variación temporal de beta. La evidencia es importante como demostración de aplicación, pero se refiere a esa muestra, esos modelos y ese criterio; no prueba superioridad universal.

### Otras aplicaciones financieras

**Relaciones variables entre activos.** Una regresión dinámica puede modelar cómo cambia la relación entre el retorno de un activo y uno o más factores. Esto puede apoyar el seguimiento de exposiciones o de razones de cobertura. Para pares de activos, una formulación de espacio de estados puede actualizar coeficientes de una relación de cobertura, siempre que la relación y el modelo de ruido estén justificados. El filtro no determina por sí solo que dos activos estén económicamente vinculados.

**Tendencia y componentes no observados.** Los modelos estructurales de series de tiempo expresan una serie como componentes latentes, por ejemplo nivel, tendencia y estacionalidad, y usan el filtro para estimarlos secuencialmente (Harvey, 1989). En finanzas, este enfoque puede aplicarse al análisis de tasas, retornos agregados o indicadores, distinguiendo una trayectoria estimada de las fluctuaciones observadas. La tendencia filtrada depende de la especificación y no debe confundirse con un precio “verdadero” ni con una señal segura de compraventa.

**Actualización de factores y exposición al riesgo.** En modelos multifactoriales, las cargas de un activo respecto de factores pueden tratarse como estados variables. De forma similar, un sistema de espacio de estados puede actualizar una relación entre tasas de interés, instrumentos y factores latentes. La utilidad práctica consiste en organizar el problema como una combinación explícita de dinámica e incertidumbre; la selección de variables y la interpretación económica siguen siendo responsabilidad del analista.

### Ventajas operativas y decisiones de modelado

- **Actualización secuencial:** integra observaciones nuevas sin volver a estimar el modelo sobre toda la historia desde cero.
- **Incertidumbre explícita:** mantiene una estimación de la covarianza del error, no solo un valor puntual.
- **Estados variables:** representa directamente coeficientes que se supone que evolucionan en el tiempo.
- **Flexibilidad estructural:** se adapta a modelos con varios estados y observaciones, siempre que la formulación sea apropiada.
- **Adaptación controlable:** la covarianza del proceso permite regular la rapidez con que cambian los estados estimados.

Estas ventajas tienen un costo de especificación. El analista debe escoger la forma de transición, las covarianzas iniciales y las matrices $Q$ y $R$. Esas elecciones afectan cuánto se suaviza la trayectoria y cuánto peso recibe cada dato. Por tanto, la conveniencia del filtro no se establece solo por la elegancia matemática: requiere diagnóstico, comparación con modelos de referencia y evaluación fuera de muestra cuando el propósito es pronosticar.

## Discusión

### Qué permite concluir la evidencia

El ejemplo de Faff et al. (2000) muestra por qué el filtro de Kalman resulta atractivo para finanzas: puede representar parámetros variables y producir estimaciones secuenciales de beta. Su resultado favorable bajo error cuadrático medio respalda la utilidad del método en una aplicación y un conjunto de datos concretos. No basta, sin embargo, para concluir que el filtro siempre pronostica mejor que GARCH, una regresión móvil u otros métodos. Los retornos financieros pueden presentar heterocedasticidad, colas pesadas, asimetría, dependencia y cambios de régimen; un modelo lineal gaussiano básico no captura automáticamente esas características.

Tampoco se debe confundir **filtrado** con **suavizado**. El filtrado estima el estado en $t$ usando información disponible hasta $t$. El suavizado puede usar observaciones posteriores para revisar una estimación pasada. En una simulación retrospectiva, utilizar suavizado o información futura puede producir una trayectoria visualmente precisa, pero sería inapropiado atribuir ese desempeño a una decisión que debía tomarse en tiempo real. Toda evaluación de una estrategia debe respetar el orden temporal de la información.

### Supuestos y límites

1. **Linealidad y distribución del ruido.** El filtro de Kalman clásico es óptimo en el marco lineal-gaussiano bajo sus supuestos. En presencia de no linealidad se usan variantes, como el filtro extendido o el de sigma points; estas no eliminan la necesidad de validar el modelo.
2. **Cambios abruptos.** Una caminata aleatoria permite evolución gradual, pero puede reaccionar tarde ante rupturas estructurales. Aumentar $Q$ acelera la respuesta y puede amplificar el ruido. Modelos de cambio de régimen o parámetros con saltos pueden ser más adecuados cuando existe evidencia de transiciones abruptas.
3. **Calibración de $Q$ y $R$.** Una especificación deficiente puede generar estimaciones inestables o excesivamente suavizadas. La inicialización del estado y su covarianza también influyen, sobre todo al inicio de la muestra.
4. **Sensibilidad a datos y frecuencia.** Datos faltantes, precios no sincronizados, valores atípicos, eventos corporativos y la elección entre frecuencia diaria o mensual afectan la inferencia. Los retornos de alta frecuencia pueden contener microestructura y ruido de negociación.
5. **Riesgo de sobreinterpretación.** Una trayectoria de beta aparentemente precisa sigue siendo una estimación con incertidumbre. No implica causalidad, no elimina el riesgo de mercado y no constituye por sí sola una recomendación de inversión.
6. **Costos y utilidad decisional.** Un menor error estadístico no garantiza mayor rentabilidad neta. Si una estimación se usa para negociar, deben considerarse costos de transacción, liquidez, restricciones, riesgo de modelo y desempeño en periodos distintos al de calibración.

### Cómo debería evaluarse una aplicación

Un estudio aplicado podría utilizar rendimientos totales ajustados de una acción o cartera y de un índice de mercado, junto con una tasa libre de riesgo coherente con la moneda y frecuencia elegidas. Debería comparar, como mínimo, una beta constante, una ventana móvil y un modelo de beta dinámica; si se estudia la volatilidad condicional, se puede incluir un enfoque GARCH. La muestra tendría que separarse cronológicamente en calibración y evaluación, evitando usar información futura.

La evaluación debería informar la especificación de transición, las covarianzas $Q$ y $R$, las condiciones iniciales, el tratamiento de datos y los criterios de comparación. Para pronóstico podrían usarse error cuadrático medio o error absoluto medio; también convendría examinar estabilidad de parámetros, residuos y desempeño durante periodos tranquilos y de tensión. Si la beta se usa en una decisión de cartera, la evaluación económica debería reportar resultados después de costos y frente a una regla de referencia. Estas son recomendaciones metodológicas para un trabajo futuro, no procedimientos ejecutados en este reporte.

### Valoración crítica

La principal contribución del filtro en finanzas es ofrecer un marco coherente para estimar variables latentes que evolucionan y cuantificar la incertidumbre de esa estimación. Su ventaja es especialmente clara cuando la pregunta de investigación tiene una interpretación dinámica concreta, como “¿cómo cambia la exposición al mercado a lo largo del tiempo?”. En cambio, aplicarlo sin una hipótesis económica, ajustar covarianzas hasta obtener una curva visualmente atractiva o elegir el modelo solo por error dentro de muestra puede dar una falsa impresión de precisión.

Por ello, la decisión no es “Kalman o ningún modelo”, sino si el mecanismo dinámico propuesto representa razonablemente el fenómeno y mejora una referencia bajo una evaluación honesta. El estudio revisado aporta evidencia favorable para una configuración y muestra particulares, pero sus propios resultados también invitan a comparar métodos y considerar combinaciones. En una aplicación profesional, la trazabilidad de los datos, la validación temporal, la sensibilidad a supuestos y la comunicación de incertidumbre son tan relevantes como la estimación recursiva.

## Conclusiones

El filtro de Kalman permite inferir recursivamente estados no observables a partir de observaciones ruidosas, mediante una etapa de predicción y otra de corrección. En finanzas, una aplicación relevante consiste en permitir que la beta de un activo cambie con el tiempo en lugar de asumir una sensibilidad constante. El marco también puede utilizarse para actualizar relaciones dinámicas entre variables y estimar componentes latentes de series, siempre que la estructura tenga justificación económica y estadística.

La evidencia de Faff et al. (2000), basada en retornos diarios de industrias del Reino Unido, reportó que el filtro con parámetros de caminata aleatoria tuvo el menor error cuadrático medio entre los modelos comparados. Ese hallazgo demuestra viabilidad, no superioridad general. El desempeño depende de la muestra, la especificación, la calibración de las covarianzas y el criterio de evaluación.

En consecuencia, el filtro es una herramienta útil para el seguimiento de parámetros financieros variables, pero no es un detector automático de cambios ni un pronosticador infalible. Sus resultados deben acompañarse de intervalos o medidas de incertidumbre, comparaciones con modelos sencillos, validación fuera de muestra y análisis de sensibilidad. Este reporte presenta conclusiones teóricas y bibliográficas; no implementa un filtro ni afirma resultados propios.

## Referencias

Durbin, J., & Koopman, S. J. (2012). *Time series analysis by state space methods* (2nd ed.). Oxford University Press.

Faff, R. W., Hillier, D., & Hillier, J. (2000). Time varying beta risk: An analysis of alternative modelling techniques. *Journal of Business Finance & Accounting, 27*(5–6), 523–554. https://doi.org/10.1111/1468-5957.00324

Harvey, A. C. (1989). *Forecasting, structural time series models and the Kalman filter*. Cambridge University Press.

Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. *Journal of Basic Engineering, 82*(1), 35–45. https://doi.org/10.1115/1.3662552

## Anexo A. División del trabajo

Completar esta tabla con los integrantes y las tareas que efectivamente realizó cada persona. Los porcentajes deben sumar 100% y no se han asignado aquí para evitar inventar contribuciones.

| Integrante | Matrícula | Actividades realizadas | Porcentaje de contribución |
|---|---|---|---:|
| [Integrante 1] | [Matrícula] | [Búsqueda bibliográfica / sección asignada] | [ % ] |
| [Integrante 2] | [Matrícula] | [Análisis / revisión de fuentes] | [ % ] |
| [Integrante 3] | [Matrícula] | [Redacción / edición / formato] | [ % ] |
| **Total** |  |  | **100%** |
