# 🎯 Visualización de Datos con matplotlib y seaborn — El arte de ver los datos antes de modelarlos

**Nivel:** 1 — Fundamentos Python para ML  
**Skill:** matplotlib + seaborn (visualización, plots básicos, EDA)  
**Datos:** Simulados — Tienda retail "MarketPro" (3 meses, 810 transacciones)

---

## 📖 Qué vas a aprender

En Machine Learning hay una regla de oro que todo practitioner aprende rápido:

> **"Siempre visualiza tus datos ANTES de modelarlos. Un gráfico te dice en 1 segundo lo que 1000 números no pueden expresar en 1 hora."**

matplotlib y seaborn son las dos librerías de visualización más importantes del ecosistema Python para ML. Se usan juntas:

| Librería | Rol | Analogía |
|----------|-----|----------|
| **matplotlib** | Motor gráfico de bajo nivel. Controlas CADA detalle | Como tener un pincel y todos los colores |
| **seaborn** | Atajos estadísticos. En 1 línea haces gráficos complejos | Como tener una plantilla profesional |

### ¿Qué gráficos aprenderás hoy?

Son los **8 tipos de gráficos esenciales** que cualquier Data Scientist usa TODOS los días:

1. **📈 Líneas** — Tendencias temporales (¿están subiendo las ventas?)
2. **📊 Barras** — Comparaciones entre categorías (¿qué categoría vende más?)
3. **📊 Histograma + KDE** — Distribuciones (¿cómo se distribuyen los precios?)
4. **📦 Boxplot** — Outliers automáticos (¿hay valores anómalos?)
5. **🔵 Scatter** — Relaciones entre variables (¿el marketing aumenta ventas?)
6. **🔥 Heatmap** — Matriz de correlación (¿qué features se relacionan?)
7. **🔗 Pairplot** — Todas las relaciones de un vistazo (el rey del EDA)
8. **🎛️ Dashboard** — Múltiples gráficos en una figura

### ¿Por qué datos de retail?

Los datos de ventas son el caso de uso MÁS COMÚN en ML empresarial:
- Tienen **patrones temporales** (fines de semana, estacionalidad)
- Tienen **correlaciones reales** (marketing → ventas)
- Tienen **categorías** (comparar grupos)
- Tienen **outliers** realistas (días festivos, promociones)
- Son **intuitivos**: entiendes el negocio sin ser experto

---

## 🧪 Código paso a paso

El proyecto completo está en `project.py`. Aquí te explico cada sección:

### 🔧 Configuración inicial

```python
sns.set_theme(style='whitegrid', palette='viridis', font_scale=1.1)
```

**¿Qué hace?** Configura el estilo GLOBAL de todos los gráficos. Con esta línea, TODOS los gráficos que hagas después se verán profesionales automáticamente. El estilo `whitegrid` da un fondo blanco con cuadrícula sutil, la paleta `viridis` es accesible para daltónicos.

### 📦 Paso 1: Generar datos realistas

Usamos numpy y pandas para crear un dataset de una tienda retail con:
- **Tendencia**: las ventas crecen un 20% en 3 meses (realista)
- **Estacionalidad**: fines de semana venden más
- **Ruido**: variabilidad aleatoria como en datos reales
- **Categorías**: Electrónica (40%), Ropa (35%), Hogar (25%)
- **Regiones**: Norte (30%), Centro (40%), Sur (30%)

### 📈 Paso 2: Gráfico de Líneas

```python
ax.plot(x, y, marker='o', markersize=3, linewidth=1.5, color='#2196F3')
```

El gráfico más básico pero poderoso. `ax.plot()` recibe coordenadas X e Y. Añadimos la **media móvil de 7 días** con `rolling(window=7).mean()` para suavizar el ruido y ver la tendencia real.

### 📊 Paso 3: Gráfico de Barras

```python
ax.bar(x, height, color=colores, edgecolor='white')
```

Para comparar categorías. Con `sns.barplot()` podemos añadir una segunda dimensión usando `hue='region'` — así vemos categoría Y región al mismo tiempo (barras agrupadas).

### 📊 Paso 4: Histograma + KDE

```python
sns.histplot(data=df, x='precio', hue='categoria', kde=True, bins=30)
```

El histograma muestra cómo se distribuyen los datos. La curva KDE (Kernel Density Estimation) suaviza el histograma en una curva continua. Con `hue` vemos la distribución separada por categoría.

### 📦 Paso 5: Boxplot

```python
sns.boxplot(data=df, x='categoria', y='ventas')
```

El boxplot muestra 5 números mágicos: mínimo, Q1 (25%), mediana (50%), Q3 (75%), máximo. Los puntos más allá de los "bigotes" son **outliers** automáticos. El método IQR (Q3 - Q1) nos da el rango intercuartil.

### 🔵 Paso 6: Scatter Plot

```python
ax.scatter(x, y, alpha=0.5, s=30, c='blue')
```

Cada punto es una observación. Muestra la RELACIÓN entre dos variables. Añadimos una **línea de regresión** con `np.polyfit()` para ver la tendencia lineal. Con `sns.scatterplot()` y `hue` coloreamos por categoría.

### 🔥 Paso 7: Heatmap de Correlación

```python
sns.heatmap(matriz_corr, annot=True, cmap='RdBu_r', center=0, vmin=-1, vmax=1)
```

**El gráfico más importante para feature selection.** Cada celda muestra la correlación de Pearson entre dos variables. Azul = correlación positiva, rojo = negativa. Las correlaciones cercanas a 1 o -1 son features prometedores para ML.

### 🔗 Paso 8: Pairplot

```python
sns.pairplot(data=df, hue='categoria', diag_kind='kde')
```

El **rey del Análisis Exploratorio**. En un solo gráfico ves:
- Diagonal: distribución de cada variable (histograma o KDE)
- Fuera de diagonal: scatter de cada par de variables
- Color: separado por categoría

### 🥧 Paso 9: Gráfico de Pastel

```python
ax.pie(values, labels=labels, autopct='%1.1f%%')
```

Solo útil con 2-5 categorías. Para reportes ejecutivos, no para análisis técnico.

### 🎛️ Paso 10: Dashboard

```python
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
```

Múltiples gráficos en una sola figura. Ideal para resúmenes ejecutivos.

---

## 📊 Resultados

### Gráficos generados

| Gráfico | Nombre | Archivo |
|---------|--------|---------|
| 📈 Líneas | Tendencia de ventas diarias | `docs/01_tendencia_ventas.png` |
| 📊 Barras | Ventas por categoría | `docs/02_ventas_por_categoria.png` |
| 📊 Barras agrupadas | Ventas categoría × región | `docs/03_ventas_categoria_region.png` |
| 📊 Histograma | Distribución de precios | `docs/04_distribucion_precios.png` |
| 📦 Boxplot | Outliers en ventas y satisfacción | `docs/05_boxplots.png` |
| 🔵 Scatter | Marketing vs Ventas | `docs/06_scatter_marketing_ventas.png` |
| 🔥 Heatmap | Matriz de correlación | `docs/07_heatmap_correlaciones.png` |
| 🔗 Pairplot | Todas las relaciones | `docs/08_pairplot.png` |
| 🥧 Pastel | Ventas por región | `docs/09_pastel_region.png` |
| 🎛️ Dashboard | Panel completo | `docs/10_dashboard_completo.png` |

### Métricas clave del dataset

```
Ventas totales: ~45,000 unidades en 3 meses
Precio promedio: ~$50 (varía por categoría y día)
Correlación marketing-ventas: ~0.65 (moderada-fuerte)
Outliers detectados: ~3-5% de los datos
```

---

## 💡 Conceptos clave

### matplotlib
- **`plt.subplots()`**: Crea figura + ejes. Es la forma moderna y recomendada de hacer gráficos
- **`ax.plot()`**, **`ax.bar()`**, **`ax.scatter()`**: Los métodos del eje para cada tipo de gráfico
- **`ax.set_title()`**, **`ax.set_xlabel()`**, **`ax.set_ylabel()`**: Personalización
- **`plt.tight_layout()`**: Ajusta automáticamente los márgenes
- **`plt.savefig()`**: Guarda como PNG (o PDF, SVG, etc.)

### seaborn
- **`sns.set_theme()`**: Configuración global de estilo
- **`sns.barplot()`**: Barras con agregación estadística automática
- **`sns.histplot()`**: Histograma + KDE en 1 línea
- **`sns.boxplot()`**: Boxplot con estilo mejorado
- **`sns.heatmap()`**: Matriz de correlación visual
- **`sns.pairplot()`**: El rey del EDA exploratorio
- **`sns.scatterplot()`**: Scatter con soporte nativo para categorías

### Principios generales
- **Siempre visualiza ANTES de modelar**: un gráfico revela problemas que las métricas esconden
- **Elige el gráfico según la pregunta**: tendencia → líneas, comparación → barras, distribución → histograma
- **Menos es más**: no sobrecargues los gráficos. Cada elemento debe añadir información
- **Color con propósito**: cada color debe significar algo (una categoría, un grupo)
- **Anotaciones**: pon los números clave en el gráfico para que sea auto-contenido

---

## 🚀 Siguiente paso

¡Felicidades! Has completado los 3 skills fundamentales de Nivel 1:
✅ pandas: DataFrames, CSV, filtros, groupby
✅ numpy: arrays, operaciones vectorizadas, broadcasting
✅ **matplotlib + seaborn: visualización ← ACABAS DE HACERLO**

### Próximo proyecto: **Data Cleaning — Limpieza de datos reales**

Aprenderás a:
- Detectar y manejar valores nulos (NaN)
- Identificar y tratar outliers
- Limpiar datos inconsistentes (formatos, strings, fechas)
- Imputar valores faltantes con diferentes estrategias
- Trabajar con datos duplicados

Este skill es el **más importante** de todos: en el mundo real, el 80% del tiempo en ML se va en limpiar datos. ¡Los modelos son la parte fácil!

---

> **"Los datos NO mienten, pero a veces te gritan la verdad en un gráfico."**
