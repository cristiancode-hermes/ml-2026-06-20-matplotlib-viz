# Learnings — 2026-06-20

## Qué se hizo
Proyecto de visualización de datos con matplotlib y seaborn. Se crearon 10 gráficos
esenciales para el análisis exploratorio de datos (EDA) sobre un dataset simulado de
una tienda retail (MarketPro) con 810 transacciones en 3 meses.

## Conceptos nuevos

- [x] matplotlib: subplots(), plot(), bar(), scatter(), hist(), pie()
- [x] seaborn: set_theme(), barplot(), histplot(), boxplot(), scatterplot(), heatmap(), pairplot()
- [x] Gráfico de líneas con media móvil (rolling window) para tendencias
- [x] Barras agrupadas con hue (doble variable categórica)
- [x] Histograma + KDE para visualizar distribuciones
- [x] Boxplot + método IQR para detección automática de outliers
- [x] Scatter plot con línea de regresión (polyfit)
- [x] Heatmap de correlación (Pearson) para feature selection
- [x] Pairplot: el rey del Análisis Exploratorio
- [x] Dashboard: múltiples subplots en una figura
- [x] Personalización: títulos, colores, leyendas, anotaciones, grid
- [x] Guardar gráficos como PNG con alta resolución

## Dificultades encontradas

- Ninguna durante la creación del proyecto. El código se ejecuta limpiamente.
- Recordatorio: usar `matplotlib.use('Agg')` para entornos sin display (servidores/CI)
- seaborn y matplotlib 3.x son totalmente compatibles con la configuración actual

## Notas para el próximo proyecto

- El próximo skill pendiente es "Data cleaning: nulos, duplicados"
- Sería ideal usar un dataset real con imperfecciones (Kaggle: Titanic, Housing, o
  mejor aún, descargar uno con datos sucios reales)
- El dataset de ventas de este proyecto podría "ensuciarse" para el próximo proyecto
  (añadir nulos, duplicados, valores inconsistentes)
- El alumno ya domina pandas, numpy, y ahora matplotlib — tiene las herramientas
  necesarias para limpiar datos reales

## Gráficos generados

| Archivo | Tipo | Descripción |
|---------|------|-------------|
| 01_tendencia_ventas.png | Líneas | Ventas diarias + media móvil 7 días |
| 02_ventas_por_categoria.png | Barras | Comparación ventas por categoría |
| 03_ventas_categoria_region.png | Barras agrupadas | Categoría × Región |
| 04_distribucion_precios.png | Histograma + KDE | Distribución de precios |
| 05_boxplots.png | Boxplot | Outliers en ventas y satisfacción |
| 06_scatter_marketing_ventas.png | Scatter | Marketing vs Ventas + regresión |
| 07_heatmap_correlaciones.png | Heatmap | Matriz de correlación |
| 08_pairplot.png | Pairplot | Todas las variables numéricas |
| 09_pastel_region.png | Pastel | Distribución por región |
| 10_dashboard_completo.png | Dashboard | Panel de 6 gráficos |
