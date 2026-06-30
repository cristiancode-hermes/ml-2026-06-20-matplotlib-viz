#!/usr/bin/env python3
"""
📊 PROYECTO: Visualización de Datos con matplotlib y seaborn
=============================================================
Nivel 1 — Skill: matplotlib + seaborn (plots básicos, visualización)

¿QUÉ VAS A APRENDER?
- A crear 8 tipos de gráficos esenciales para ML
- Cuándo usar cada tipo de gráfico (y por qué)
- Personalizar: colores, tamaños, títulos, leyendas, anotaciones
- Subplots: múltiples gráficos en una sola figura
- seaborn: gráficos estadísticos con 1 línea de código
- Guardar gráficos como PNG para informes y presentaciones

DATOS SIMULADOS: Tienda retail "MarketPro" — 3 meses de operaciones
- Ventas diarias, clientes, productos, marketing, satisfacción
- 3 categorías de productos: Electrónica, Ropa, Hogar
- 3 regiones: Norte, Centro, Sur
- Datos con patrones realistas (findesemana altos, correlaciones, outliers)

FRASE CLAVE: "Siempre visualiza tus datos ANTES de modelarlos.
              Un gráfico te dice en 1 segundo lo que 1000 números no pueden."
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Para entornos sin pantalla (servidores)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')  # Limpiamos la salida de advertencias

# =====================================================================
# CONFIGURACIÓN INICIAL
# =====================================================================
# Creamos la carpeta donde se guardarán TODOS los gráficos
os.makedirs('docs', exist_ok=True)

# Configuramos el estilo de seaborn para que TODOS los gráficos se vean profesionales
# Esto afecta globalmente a matplotlib también
sns.set_theme(
    style='whitegrid',        # Fondo blanco con cuadrícula sutil
    palette='viridis',        # Paleta de colores profesional (accesible para daltónicos)
    font='DejaVu Sans',       # Fuente limpia y legible
    font_scale=1.1,           # Texto un 10% más grande que el default
    rc={
        'figure.figsize': (12, 6),   # Tamaño por defecto de las figuras
        'figure.dpi': 120,           # Resolución alta para que se vea nítido
        'savefig.dpi': 150,           # Resolución al guardar PNG
        'savefig.bbox': 'tight',     # No cortar bordes al guardar
    }
)

# Colores personalizados que usaremos en varios gráficos
COLORES = {
    'electronica': '#2196F3',  # Azul
    'ropa': '#FF5722',         # Naranja
    'hogar': '#4CAF50',        # Verde
    'norte': '#E91E63',        # Rosa
    'centro': '#9C27B0',       # Púrpura
    'sur': '#00BCD4',          # Cian
    'marketing': '#FF9800',    # Ámbar
}

print("=" * 72)
print("📊  MATPLOTLIB + SEABORN: EL ARTE DE VISUALIZAR DATOS")
print("=" * 72)
print("""
🧠 ¿Por qué aprender visualización ANTES del Machine Learning?

   En ML, tú no le "dices" al modelo qué patrones buscar.
   El modelo los encuentra SOLO. Pero TÚ necesitas:
   
   1. Entender tus datos antes de meterlos al modelo
   2. Detectar problemas: outliers, datos faltantes, correlaciones falsas
   3. Comunicar resultados: un gráfico vale más que 1000 métricas
   4. Debuggear modelos: ¿por qué falla? → mira los residuos

   matplotlib → el "motor gráfico" de Python (control TOTAL)
   seaborn   → atajos estadísticos (1 línea = gráfico complejo)
   
   ¡Se usan JUNTOS! seaborn usa matplotlib por debajo.
""")

# =====================================================================
# PASO 1: CREAR DATOS REALISTAS CON NUMPY + PANDAS
# =====================================================================
print("\n" + "=" * 72)
print("📦 PASO 1: Generando datos realistas de una tienda retail")
print("=" * 72)

# ----------------------------------------------------------------------
# 1.1: Generamos fechas para ~3 meses (90 días)
# ----------------------------------------------------------------------
# Usamos pd.date_range() — crea un rango de fechas.
# 'freq=D' significa frecuencia diaria. 90 días partiendo del 1 de abril.
fechas = pd.date_range(start='2026-04-01', periods=90, freq='D')
print(f"   ✅ Rango de fechas: {fechas[0].date()} → {fechas[-1].date()} ({len(fechas)} días)")

# ----------------------------------------------------------------------
# 1.2: Variables base con numpy — patrones realistas
# ----------------------------------------------------------------------
np.random.seed(42)  # Semilla fija para que los resultados sean REPRODUCIBLES

n_dias = len(fechas)

# --- Tendencia base de ventas ---
# Las ventas en retail tienden a CRECER mes a mes.
# Creamos una tendencia lineal ascendente del 20% en 3 meses.
# np.linspace(1.0, 1.2, n_dias) = valores desde 1.0 hasta 1.2 en 90 pasos
tendencia = np.linspace(1.0, 1.2, n_dias)

# --- Efecto día de la semana ---
# Los findes (viernes-sábado) se vende más que entre semana.
# .dayofweek() devuelve: lunes=0, martes=1, ..., domingo=6
# Creamos un array con factores multiplicadores según el día.
dia_semana = fechas.dayofweek  # 0=lunes, 1=martes, ..., 6=domingo

# Factor de ventas por día: jueves(3)=1.2, viernes(4)=1.5, sábado(5)=1.6, domingo(6)=1.0
factor_dia = np.where(dia_semana == 4, 1.5,     # viernes: +50%
             np.where(dia_semana == 5, 1.6,     # sábado: +60%
             np.where(dia_semana == 3, 1.2,     # jueves: +20%
             np.where(dia_semana == 6, 1.0,     # domingo: normal (baja)
                     0.8))))                    # lunes a miércoles: -20%
print(f"   ✅ Factor día de semana creado: lunes={factor_dia[0]:.1f}, viernes={factor_dia[3]:.1f}, sábado={factor_dia[4]:.1f}")

# --- Ruido aleatorio ---
# En datos reales SIEMPRE hay ruido. Lo simulamos con distribución normal.
# np.random.normal(media, desviación, tamaño)
# Las ventas reales tienen variabilidad natural.
ruido_ventas = np.random.normal(1.0, 0.08, n_dias)
ruido_clientes = np.random.normal(1.0, 0.10, n_dias)

# --- Ventas base ---
# Combinamos tendencia + día + ruido
# 200 unidades/día es la base, luego la ajustamos
ventas_base = 200 * tendencia * factor_dia * ruido_ventas
ventas_base = np.clip(ventas_base, 50, None)  # Mínimo 50 ventas/día
ventas = ventas_base.astype(int)

# --- Clientes únicos ---
# Correlacionado con ventas (más ventas → más clientes)
# Pero con más variabilidad
clientes = (150 * tendencia * factor_dia * ruido_clientes).astype(int)
clientes = np.clip(clientes, 30, None)

# --- Precio promedio ---
# Varía según el día (los findes hay promociones)
precio_base = np.random.normal(45, 8, n_dias)
# Los jueves ("jueves de descuento") bajan precios
precio_base[dia_semana == 3] *= 0.85
# Los domingos también
precio_base[dia_semana == 6] *= 0.90
precio_promedio = np.round(precio_base, 1)

# --- Inversión en marketing ---
# Los fines de semana se invierte más en marketing
marketing = np.where(dia_semana >= 4,  # jueves a domingo
                     np.random.uniform(800, 1500, n_dias),
                     np.random.uniform(300, 800, n_dias))
marketing = np.round(marketing, 0)

# --- Satisfacción del cliente (1-5) ---
# Cuando hay más ventas, la satisfacción tiende a bajar (más espera)
satisfaccion_base = np.random.normal(4.0, 0.3, n_dias)
# Ajuste: más ventas → menor satisfacción
satisfaccion_base = satisfaccion_base - 0.3 * (ventas / ventas.mean() - 1)
satisfaccion = np.clip(np.round(satisfaccion_base, 1), 1.0, 5.0)

# --- Devoluciones ---
# Aproximadamente 3-8% de las ventas se devuelven
tasa_devolucion = np.random.uniform(0.03, 0.08, n_dias)
devoluciones = (ventas * tasa_devolucion).astype(int)

print("   ✅ Variables numéricas generadas con patrones realistas")

# ----------------------------------------------------------------------
# 1.3: Asignamos categorías y regiones (datos categóricos)
# ----------------------------------------------------------------------
# A CADA DÍA le asignamos una categoría y región principal
# Pero en realidad cada día tiene TODAS las categorías activas.
# Vamos a hacer un dataset más rico: expandimos a nivel producto-región.

# Enfoque: creamos un DataFrame donde CADA FILA es una combinación
# día × categoría × región. Así tenemos datos más realistas.
categorias = ['Electrónica', 'Ropa', 'Hogar']
regiones = ['Norte', 'Centro', 'Sur']

# Creamos todas las combinaciones: 90 días × 3 categorías × 3 regiones = 810 filas
filas = []
for i, fecha in enumerate(fechas):
    for cat in categorias:
        for reg in regiones:
            # Cada categoría-región tiene una porción de las ventas totales del día
            # Electrónica: ~40%, Ropa: ~35%, Hogar: ~25%
            if cat == 'Electrónica':
                porc_cat = 0.40 + np.random.uniform(-0.05, 0.05)
            elif cat == 'Ropa':
                porc_cat = 0.35 + np.random.uniform(-0.05, 0.05)
            else:  # Hogar
                porc_cat = 0.25 + np.random.uniform(-0.05, 0.05)

            # Cada región tiene participación similar con variación
            if reg == 'Norte':
                porc_reg = 0.30 + np.random.uniform(-0.03, 0.03)
            elif reg == 'Centro':
                porc_reg = 0.40 + np.random.uniform(-0.03, 0.03)  # Centro es más grande
            else:  # Sur
                porc_reg = 0.30 + np.random.uniform(-0.03, 0.03)

            # Ventas para esta combinación
            ventas_combo = int(ventas[i] * porc_cat * porc_reg)

            # Marketing asignado proporcionalmente
            mkt_combo = marketing[i] * porc_cat * porc_reg

            # Precio varía por categoría
            if cat == 'Electrónica':
                precio_combo = precio_promedio[i] * np.random.uniform(1.5, 2.5)
            elif cat == 'Ropa':
                precio_combo = precio_promedio[i] * np.random.uniform(0.4, 0.8)
            else:
                precio_combo = precio_promedio[i] * np.random.uniform(0.6, 1.2)

            filas.append({
                'fecha': fecha,
                'categoria': cat,
                'region': reg,
                'ventas': max(ventas_combo, 0),
                'precio_promedio': round(precio_combo, 1),
                'marketing': round(mkt_combo, 0),
                'satisfaccion': satisfaccion[i],
                'devoluciones': max(int(devoluciones[i] * porc_cat * porc_reg * 3), 0),
                'dia_semana': fecha.day_name()  # 'Monday', 'Tuesday', etc.
            })

# Creamos el DataFrame
df = pd.DataFrame(filas)

# Añadimos información temporal útil
df['mes'] = df['fecha'].dt.month_name()
df['semana'] = df['fecha'].dt.isocalendar().week.astype(int)
df['dia_num'] = df['fecha'].dt.dayofweek  # 0=lunes, ..., 6=domingo

print(f"   ✅ DataFrame creado: {df.shape[0]:,} filas × {df.shape[1]} columnas")
print(f"   📋 Columnas: {', '.join(df.columns.tolist())}")
print(f"\n   📊 Primeras 5 filas del dataset:")
print(df.head().to_string())
print(f"\n   📊 Estadísticas básicas:")
print(df[['ventas', 'precio_promedio', 'marketing', 'satisfaccion', 'devoluciones']].describe().to_string())

# =====================================================================
# PASO 2: GRÁFICO DE LÍNEAS — Tendencias temporales
# =====================================================================
print("\n" + "=" * 72)
print("📈 PASO 2: Gráfico de Líneas — ¿Cómo evolucionan las ventas en el tiempo?")
print("=" * 72)
print("""
   📍 ¿Para qué sirve?
      Ver TENDENCIAS: ¿están subiendo o bajando las ventas?
      Ver ESTACIONALIDAD: ¿hay patrones que se repiten?
      Ver ANOMALÍAS: ¿hubo un día inusualmente malo/bueno?
   
   📍 ¿Cuándo usarlo en ML?
      - Antes de modelos de series temporales (ARIMA, LSTM)
      - Para ver si los datos necesitan diferenciación
      - Detectar cambios de régimen (el comportamiento cambió)
""")

# ----------------------------------------------------------------------
# 2.1: Agrupamos ventas por fecha (todas las categorías y regiones juntas)
# ----------------------------------------------------------------------
# En ML es muy común tener que AGRUPAR datos antes de graficar.
# groupby + sum = total por día
ventas_diarias = df.groupby('fecha')['ventas'].sum()

# Creamos la figura — el "lienzo" donde pintamos
fig, ax = plt.subplots(figsize=(14, 6))

# ----------------------------------------------------------------------
# 2.2: El gráfico de líneas principal
# ----------------------------------------------------------------------
# ax.plot() es el corazón de matplotlib para líneas
# Parámetros:
#   x = fechas (eje horizontal)
#   y = ventas (eje vertical)
#   marker='o' = poner un punto en cada dato
#   markersize=3 = tamaño de los puntos
#   linewidth=1.5 = grosor de la línea
#   color=... = color HEX o nombre
#   alpha=0.8 = transparencia (0=invisible, 1=sólido)
ax.plot(
    ventas_diarias.index,     # Eje X: las fechas
    ventas_diarias.values,    # Eje Y: las ventas
    marker='o',               # Pone un circulito en cada día
    markersize=3,             # Tamaño del circulito
    linewidth=1.5,            # Grosor de la línea
    color='#2196F3',          # Azul material design
    alpha=0.8,                # Un poquito transparente
    label='Ventas diarias'    # Etiqueta para la leyenda
)

# ----------------------------------------------------------------------
# 2.3: Línea de tendencia — suaviza el ruido para ver el patrón general
# ----------------------------------------------------------------------
# Calculamos el promedio móvil de 7 días (ventana de una semana)
# Esto suaviza el ruido de días individuales y muestra la tendencia real
# .rolling(window=7).mean() = para cada día, promedia los 7 días anteriores
tendencia_7d = ventas_diarias.rolling(window=7).mean()

ax.plot(
    tendencia_7d.index,
    tendencia_7d.values,
    color='#FF5722',           # Naranja — línea de tendencia
    linewidth=3,               # Más gruesa para destacar
    linestyle='--',            # Línea punteada (--), también: '-', ':', '-.'
    alpha=0.9,
    label='Tendencia (media 7 días)'
)

# ----------------------------------------------------------------------
# 2.4: Personalización — esto es lo que hace que un gráfico se vea profesional
# ----------------------------------------------------------------------
ax.set_title(
    '📈 Ventas Diarias — MarketPro (Abril a Junio 2026)',
    fontsize=16,
    fontweight='bold',
    pad=15  # padding: espacio entre título y gráfico
)
ax.set_xlabel('Fecha', fontsize=12, fontweight='bold')
ax.set_ylabel('Ventas (unidades)', fontsize=12, fontweight='bold')

# Leyenda: explica qué significa cada línea/color
ax.legend(loc='upper left', fontsize=11, frameon=True, fancybox=True, shadow=True)

# Cuadrícula: ayuda a leer valores
ax.grid(True, alpha=0.3, linestyle='--')

# Formato del eje X: rotamos las fechas para que se lean bien
plt.xticks(rotation=45, ha='right')

# Ajustamos los márgenes para que nada se corte
plt.tight_layout()

# Guardamos como PNG — importante para compartir en informes
plt.savefig('docs/01_tendencia_ventas.png')
plt.close()  # Cerramos la figura para liberar memoria
print("   ✅ Gráfico 1 guardado: docs/01_tendencia_ventas.png")

# =====================================================================
# PASO 3: GRÁFICO DE BARRAS — Comparaciones entre categorías
# =====================================================================
print("\n" + "=" * 72)
print("📊 PASO 3: Gráfico de Barras — ¿Qué categoría vende más?")
print("=" * 72)
print("""
   📍 ¿Para qué sirve?
      Comparar valores entre categorías de forma VISUAL e INMEDIATA
      Ver qué categoría domina, cuál se queda atrás
   
   📍 ¿Cuándo usarlo en ML?
      - Análisis exploratorio: ¿qué features numéricos varían por categoría?
      - Para decidir si una variable categórica es predictiva
      - Para comunicar resultados a negocios
""")

# ----------------------------------------------------------------------
# 3.1: Agrupamos por categoría
# ----------------------------------------------------------------------
ventas_cat = df.groupby('categoria')['ventas'].sum().sort_values(ascending=False)
print(f"\n   Ventas totales por categoría:")
for cat, val in ventas_cat.items():
    print(f"      {cat}: {val:,.0f} unidades")

# Colores para cada categoría
colores_cat = [COLORES['electronica'], COLORES['ropa'], COLORES['hogar']]

fig, ax = plt.subplots(figsize=(10, 6))

# ----------------------------------------------------------------------
# 3.2: El gráfico de barras
# ----------------------------------------------------------------------
# ax.bar() = barras verticales
#   x = nombres de categorías (índice)
#   height = valores (altura de cada barra)
#   color = colores para cada barra (lista)
#   edgecolor = color del borde de cada barra
#   linewidth = grosor del borde
barras = ax.bar(
    ventas_cat.index,         # Las 3 categorías en el eje X
    ventas_cat.values,        # Ventas totales en el eje Y
    color=colores_cat,        # Un color distinto por categoría
    edgecolor='white',        # Borde blanco para separar visualmente
    linewidth=1.5,
    width=0.6,                # Ancho de las barras (0-1)
    alpha=0.9
)

# ----------------------------------------------------------------------
# 3.3: Anotaciones — poner el VALOR EXACTO encima de cada barra
# ----------------------------------------------------------------------
# Esto es clave para que el gráfico sea COMPLETO (no obliga a adivinar)
for barra in barras:
    altura = barra.get_height()
    ax.annotate(
        f'{altura:,.0f}',                    # Texto: el número con separador de miles
        xy=(barra.get_x() + barra.get_width() / 2, altura),  # Posición: centro de la barra
        xytext=(0, 5),                        # Desplazamiento: 5px hacia arriba
        textcoords='offset points',           # Las coordenadas de xytext son en puntos
        ha='center',                          # Alineación horizontal: centro
        fontsize=13,
        fontweight='bold'
    )

# Personalización
ax.set_title('🛒 Ventas Totales por Categoría', fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('Categoría', fontsize=12, fontweight='bold')
ax.set_ylabel('Ventas Totales (unidades)', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')  # Solo líneas horizontales

plt.tight_layout()
plt.savefig('docs/02_ventas_por_categoria.png')
plt.close()
print("   ✅ Gráfico 2 guardado: docs/02_ventas_por_categoria.png")

# ----------------------------------------------------------------------
# 3.4: Barras AGRUPADAS — categoría × región
# ----------------------------------------------------------------------
# Este es el gráfico de barras más útil en análisis de datos:
# Comparar DOS variables categóricas a la vez.
print("\n   📊 Ventas por categoría Y región (barras agrupadas):")

# Creamos una tabla pivote: filas=categoría, columnas=región, valores=ventas
# .pivot_table() es UNA de las funciones más útiles de pandas para ML
ventas_cat_reg = df.pivot_table(
    values='ventas',
    index='categoria',
    columns='region',
    aggfunc='sum'
)
print("\n   Tabla pivote (categoría × región):")
print(ventas_cat_reg.to_string())

fig, ax = plt.subplots(figsize=(11, 7))

# seaborn tiene una función especial para barras agrupadas
# sns.barplot() hace todo el cálculo automáticamente
# Datos en formato LARGO (tidy): cada fila es una observación
sns.barplot(
    data=df,                    # DataFrame completo
    x='categoria',              # Eje X: categorías
    y='ventas',                 # Eje Y: ventas
    hue='region',               # Color por región (crea grupos)
    palette=[COLORES['norte'], COLORES['centro'], COLORES['sur']],  # Colores personalizados
    estimator=sum,              # Agregación: suma (default: media)
    errorbar=None,              # Sin barras de error
    edgecolor='white',
    linewidth=1,
    ax=ax
)

ax.set_title('🏪 Ventas por Categoría y Región', fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('Categoría', fontsize=12, fontweight='bold')
ax.set_ylabel('Ventas Totales', fontsize=12, fontweight='bold')
ax.legend(title='Región', title_fontsize=11, fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('docs/03_ventas_categoria_region.png')
plt.close()
print("   ✅ Gráfico 3 guardado: docs/03_ventas_categoria_region.png")

# =====================================================================
# PASO 4: HISTOGRAMA + KDE — Distribuciones de datos
# =====================================================================
print("\n" + "=" * 72)
print("📊 PASO 4: Histograma — ¿Cómo se distribuyen los precios?")
print("=" * 72)
print("""
   📍 ¿Para qué sirve?
      Ver la FORMA de los datos: ¿son normales? ¿sesgados? ¿bimodales?
      Detectar OUTLIERS: valores que se salen de lo normal
      Entender los RANGOS típicos de cada variable
   
   📍 ¿Cuándo usarlo en ML?
      - Antes de escalar features (¿necesito StandardScaler o RobustScaler?)
      - Para ver si una variable está sesgada (log-transform?)
      - Para detectar problemas de calidad de datos
""")

fig, axes = plt.subplots(2, 1, figsize=(14, 10))
# axes es un array con [ax0, ax1] — el primer y segundo subplot

# ----------------------------------------------------------------------
# 4.1: Histograma tradicional (eje de frecuencias)
# ----------------------------------------------------------------------
# ax.hist() cuenta cuántos valores caen en cada "cajón" (bin)
# bins=30 → divide el rango en 30 intervalos iguales
axes[0].hist(
    df['precio_promedio'],     # Datos: precios de todos los productos
    bins=30,                    # Número de cajones
    color='#4CAF50',            # Verde
    edgecolor='white',          # Borde blanco
    alpha=0.8,
    density=False               # Mostrar frecuencias absolutas (no densidad)
)
axes[0].set_title('💰 Distribución de Precios (Histograma)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Precio Promedio ($)', fontsize=11)
axes[0].set_ylabel('Frecuencia (# de observaciones)', fontsize=11)
axes[0].grid(alpha=0.3, linestyle='--')

# Añadimos líneas verticales para la media y mediana
media_precio = df['precio_promedio'].mean()
mediana_precio = df['precio_promedio'].median()
axes[0].axvline(media_precio, color='red', linestyle='--', linewidth=2,
                label=f'Media: ${media_precio:.1f}')
axes[0].axvline(mediana_precio, color='darkorange', linestyle=':', linewidth=2,
                label=f'Mediana: ${mediana_precio:.1f}')
axes[0].legend(fontsize=10)

# ----------------------------------------------------------------------
# 4.2: Histograma + KDE (la forma favorita en ML)
# ----------------------------------------------------------------------
# KDE = Kernel Density Estimation → suaviza el histograma en una curva
# sns.histplot() combina histograma + KDE en 1 sola función
sns.histplot(
    data=df,
    x='precio_promedio',
    hue='categoria',            # Un color por categoría (apilado o separado)
    bins=30,
    kde=True,                   # ← Activa la curva de densidad KDE
    alpha=0.5,
    palette=[COLORES['electronica'], COLORES['ropa'], COLORES['hogar']],
    ax=axes[1]
)
axes[1].set_title('💰 Distribución de Precios por Categoría', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Precio Promedio ($)', fontsize=11)
axes[1].set_ylabel('Densidad', fontsize=11)
axes[1].grid(alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('docs/04_distribucion_precios.png')
plt.close()
print("   ✅ Gráfico 4 guardado: docs/04_distribucion_precios.png")
print(f"   📊 Precio promedio: media=${media_precio:.1f}, mediana=${mediana_precio:.1f}")

# =====================================================================
# PASO 5: BOXPLOT — Detección de Outliers
# =====================================================================
print("\n" + "=" * 72)
print("📦 PASO 5: Boxplot — Detectando outliers como un profesional")
print("=" * 72)
print("""
   📍 ¿Para qué sirve?
      Visualizar distribución con 5 números: min, Q1, mediana, Q3, max
      Detectar OUTLIERS de forma automática (puntos fuera de los bigotes)
      Comparar múltiples grupos en un solo gráfico
   
   📍 Anatomía de un boxplot:
      
        ┌────────────────────────────┐
        │          OUTLIER (○)       │ ← valores atípicos
        │    ─── bigote superior     │ ← Q3 + 1.5×RIQ
        │  ┌─────────────────────┐   │
        │  │    Q3 (75%)         │   │
        │  │    MEDIANA (─)      │   │ ← la línea del medio
        │  │    Q1 (25%)         │   │
        │  └─────────────────────┘   │
        │    ─── bigote inferior     │ ← Q1 - 1.5×RIQ
        │          OUTLIER (○)       │
        └────────────────────────────┘
      RIQ = Rango Intercuartil = Q3 - Q1
""")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# ----------------------------------------------------------------------
# 5.1: Boxplot simple — ventas por categoría
# ----------------------------------------------------------------------
# sns.boxplot() = boxplot con estilo estadístico mejorado
sns.boxplot(
    data=df,
    x='categoria',
    y='ventas',
    palette=[COLORES['electronica'], COLORES['ropa'], COLORES['hogar']],
    ax=axes[0],
    width=0.5,
    flierprops={                # Personalizamos cómo se ven los outliers
        'marker': 'o',
        'markerfacecolor': 'red',
        'markersize': 5,
        'alpha': 0.6
    }
)
axes[0].set_title('📦 Ventas por Categoría', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Categoría', fontsize=11)
axes[0].set_ylabel('Ventas (unidades)', fontsize=11)
axes[0].grid(axis='y', alpha=0.3, linestyle='--')

# ----------------------------------------------------------------------
# 5.2: Boxplot — satisfacción por día de la semana
# ----------------------------------------------------------------------
# Ordenamos los días de la semana correctamente (no alfabético)
orden_dias = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
# Creamos etiquetas en español
dias_es = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']

sns.boxplot(
    data=df,
    x='dia_semana',
    y='satisfaccion',
    order=orden_dias,           # Orden personalizado
    palette='viridis',
    ax=axes[1],
    width=0.5,
    flierprops={
        'marker': 'o',
        'markerfacecolor': 'red',
        'markersize': 5,
        'alpha': 0.6
    }
)
axes[1].set_title('😊 Satisfacción por Día de la Semana', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Día de la Semana', fontsize=11)
# Cambiamos las etiquetas a español
axes[1].set_xticklabels(dias_es, rotation=30)
axes[1].set_ylabel('Satisfacción (1-5)', fontsize=11)
axes[1].grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('docs/05_boxplots.png')
plt.close()
print("   ✅ Gráfico 5 guardado: docs/05_boxplots.png")

# ----------------------------------------------------------------------
# 5.3: Analicemos los outliers
# ----------------------------------------------------------------------
# Identificamos outliers usando el método IQR (Rango Intercuartil)
Q1 = df['ventas'].quantile(0.25)
Q3 = df['ventas'].quantile(0.75)
IQR = Q3 - Q1
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

outliers = df[(df['ventas'] < limite_inferior) | (df['ventas'] > limite_superior)]
print(f"\n   📊 Análisis de outliers en ventas:")
print(f"      Q1 (25%): {Q1:.0f}")
print(f"      Q3 (75%): {Q3:.0f}")
print(f"      RIQ: {IQR:.0f}")
print(f"      Bigote inferior: {limite_inferior:.0f}")
print(f"      Bigote superior: {limite_superior:.0f}")
print(f"      Outliers detectados: {len(outliers)} de {len(df)} ({len(outliers)/len(df)*100:.1f}%)")

if len(outliers) > 0:
    print(f"\n      Top 5 outliers:")
    print(outliers[['fecha', 'categoria', 'region', 'ventas']].head(5).to_string())

# =====================================================================
# PASO 6: SCATTER PLOT — Relaciones entre variables
# =====================================================================
print("\n" + "=" * 72)
print("🔵 PASO 6: Scatter Plot — ¿El marketing aumenta las ventas?")
print("=" * 72)
print("""
   📍 ¿Para qué sirve?
      Ver la RELACIÓN entre DOS variables numéricas
      Identificar correlaciones (positiva, negativa, nula)
      Detectar patrones no lineales y clusters
   
   📍 ¿Cuándo usarlo en ML?
      - Feature selection: ¿qué variables se relacionan con el target?
      - Multicolinealidad: ¿dos features están muy correlacionados?
      - Visualizar clusters antes de K-Means
""")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# ----------------------------------------------------------------------
# 6.1: Scatter simple — Marketing vs Ventas
# ----------------------------------------------------------------------
# Para ver la relación GENERAL, sin separar por categoría
axes[0].scatter(
    df['marketing'],            # Eje X: inversión en marketing
    df['ventas'],               # Eje Y: ventas
    alpha=0.5,                  # Transparencia para ver densidad
    s=30,                       # Tamaño de los puntos
    c='#2196F3',                # Color azul
    edgecolors='white',         # Borde blanco para separar puntos
    linewidth=0.3
)
axes[0].set_title('📢 Marketing vs Ventas', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Inversión en Marketing ($)', fontsize=11)
axes[0].set_ylabel('Ventas (unidades)', fontsize=11)
axes[0].grid(alpha=0.3, linestyle='--')

# Añadimos línea de regresión (tendencia lineal) con numpy
# polyfit calcula los coeficientes de la recta: y = m*x + b
# poly1d crea la función polinómica
m, b = np.polyfit(df['marketing'], df['ventas'], 1)
x_line = np.linspace(df['marketing'].min(), df['marketing'].max(), 100)
axes[0].plot(x_line, m * x_line + b, color='red', linewidth=2, linestyle='--',
             label=f'Regresión lineal (m={m:.2f})')
axes[0].legend(fontsize=10)

# ----------------------------------------------------------------------
# 6.2: Scatter con COLOR por categoría — el más informativo
# ----------------------------------------------------------------------
# sns.scatterplot() automáticamente asigna colores por una variable categórica
sns.scatterplot(
    data=df,
    x='marketing',
    y='ventas',
    hue='categoria',            # Color según la categoría
    style='categoria',          # Forma del punto según la categoría
    palette=[COLORES['electronica'], COLORES['ropa'], COLORES['hogar']],
    alpha=0.7,
    s=50,
    ax=axes[1]
)
axes[1].set_title('📢 Marketing vs Ventas (por Categoría)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Inversión en Marketing ($)', fontsize=11)
axes[1].set_ylabel('Ventas (unidades)', fontsize=11)
axes[1].legend(title='Categoría', title_fontsize=11, fontsize=10)
axes[1].grid(alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('docs/06_scatter_marketing_ventas.png')
plt.close()
print("   ✅ Gráfico 6 guardado: docs/06_scatter_marketing_ventas.png")

# Calculamos la correlación de Pearson
corr_marketing_ventas = df['marketing'].corr(df['ventas'])
print(f"\n   📊 Correlación marketing ↔ ventas: {corr_marketing_ventas:.3f}")
print(f"      Interpretación: ", end="")
if abs(corr_marketing_ventas) > 0.7:
    print("Correlación FUERTE")
elif abs(corr_marketing_ventas) > 0.4:
    print("Correlación MODERADA")
else:
    print("Correlación DÉBIL")

# =====================================================================
# PASO 7: HEATMAP — Matriz de correlaciones
# =====================================================================
print("\n" + "=" * 72)
print("🔥 PASO 7: Heatmap — El mapa del tesoro de las correlaciones")
print("=" * 72)
print("""
   📍 ¿Para qué sirve?
      Ver TODAS las correlaciones entre TODAS las variables numéricas
      Identificar qué features se relacionan con el target
      Detectar multicolinealidad (features que se parecen mucho)
   
   📍 ¿Cuándo usarlo en ML?
      - Feature selection: quedarse con las variables que más correlacionan con Y
      - Antes de regresión lineal: evitar features correlacionados entre sí
      - Reducción de dimensionalidad: features redundantes → PCA
""")

# ----------------------------------------------------------------------
# 7.1: Calculamos la matriz de correlación
# ----------------------------------------------------------------------
# Seleccionamos solo las columnas numéricas
# .corr() calcula Pearson por defecto (también: spearman, kendall)
cols_numericas = ['ventas', 'precio_promedio', 'marketing', 'satisfaccion', 'devoluciones']
matriz_corr = df[cols_numericas].corr()

print("\n   📊 Matriz de correlación (Pearson):")
print(matriz_corr.to_string())

fig, ax = plt.subplots(figsize=(10, 8))

# ----------------------------------------------------------------------
# 7.2: Heatmap con seaborn
# ----------------------------------------------------------------------
# sns.heatmap() es el estándar para matrices de correlación en ML
sns.heatmap(
    matriz_corr,                # La matriz de correlación
    annot=True,                 # Mostrar los valores numéricos dentro de cada celda
    fmt='.2f',                  # Formato: 2 decimales
    cmap='RdBu_r',              # Mapa de colores: rojo=negativa, azul=positiva
    center=0,                   # Centro del mapa de color en 0
    vmin=-1,                    # Valor mínimo de correlación
    vmax=1,                     # Valor máximo de correlación
    square=True,                # Celdas cuadradas (no rectangulares)
    linewidths=0.8,             # Líneas entre celdas
    linecolor='white',
    cbar_kws={
        'shrink': 0.8,          # Encoger la barra de color
        'label': 'Correlación de Pearson'
    },
    annot_kws={
        'fontsize': 14,
        'fontweight': 'bold'
    },
    ax=ax
)

ax.set_title('🔥 Matriz de Correlación — Variables Numéricas\n(Más azul = correlación positiva, más rojo = negativa)',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=11)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)

plt.tight_layout()
plt.savefig('docs/07_heatmap_correlaciones.png')
plt.close()
print("   ✅ Gráfico 7 guardado: docs/07_heatmap_correlaciones.png")

# ----------------------------------------------------------------------
# 7.3: Analizamos las correlaciones más importantes
# ----------------------------------------------------------------------
# Ordenamos las correlaciones con 'ventas' (nuestro target potencial)
corr_con_ventas = matriz_corr['ventas'].drop('ventas').sort_values(ascending=False)
print("\n   📊 Correlaciones con VENTAS (nuestro target potencial):")
for var, val in corr_con_ventas.items():
    print(f"      {var}: {val:+.3f}")

# =====================================================================
# PASO 8: PAIRPLOT — Todas las variables, todas las relaciones
# =====================================================================
print("\n" + "=" * 72)
print("🔗 PASO 8: Pairplot — Un vistazo completo a TODAS las relaciones")
print("=" * 72)
print("""
   📍 ¿Para qué sirve?
      Es como tener 10 scatter plots + 4 histogramas en UN solo gráfico
      La diagonal muestra la distribución de cada variable
      Fuera de la diagonal: scatter plot de cada par de variables
      Color por categoría para ver patrones por grupo
   
   📍 ¿Cuándo usarlo en ML?
      - Primer vistazo a un dataset nuevo (es lo PRIMERO que hago)
      - Identificar relaciones no lineales
      - Ver si los clusters tienen sentido
      Dato: en kaggle, el pairplot es el gráfico más usado en EDA
""")

# ----------------------------------------------------------------------
# 8.1: Pairplot con seaborn
# ----------------------------------------------------------------------
# sns.pairplot() = EL atajo definitivo para EDA visual
# Con hue='categoria' colorea cada punto por categoría
# Esto es INCREÍBLEMENTE informativo
g = sns.pairplot(
    data=df[cols_numericas + ['categoria']],  # Solo numéricas + categoría
    hue='categoria',                            # Color por categoría
    palette=[COLORES['electronica'], COLORES['ropa'], COLORES['hogar']],
    diag_kind='kde',                            # En la diagonal: curvas KDE
    plot_kws={                                  # Personalización de los scatters
        'alpha': 0.5,
        's': 20,
        'edgecolor': 'white',
        'linewidth': 0.3
    },
    diag_kws={                                  # Personalización de las KDE
        'alpha': 0.6,
        'fill': True
    },
    height=2.5                                  # Tamaño de cada subplot
)

# Ajustamos el título general
g.fig.suptitle('🔗 Pairplot: Relaciones entre todas las variables numéricas',
               fontsize=16, fontweight='bold', y=1.02)
g.fig.subplots_adjust(top=0.95)

# Guardamos
plt.savefig('docs/08_pairplot.png', bbox_inches='tight')
plt.close()
print("   ✅ Gráfico 8 guardado: docs/08_pairplot.png")

# =====================================================================
# PASO 9: GRÁFICO DE PASTEL — Distribución de ventas por región
# =====================================================================
print("\n" + "=" * 72)
print("🥧 PASO 9: Gráfico de Pastel — ¿Qué región aporta más?")
print("=" * 72)

# ----------------------------------------------------------------------
# 9.1: Agrupamos ventas por región
# ----------------------------------------------------------------------
ventas_region = df.groupby('region')['ventas'].sum()
print(f"\n   Ventas totales por región:")
for reg, val in ventas_region.items():
    print(f"      {reg}: {val:,.0f} ({val/ventas_region.sum()*100:.1f}%)")

fig, ax = plt.subplots(figsize=(8, 8))

# ----------------------------------------------------------------------
# 9.2: Gráfico de pastel (pie chart)
# ----------------------------------------------------------------------
# IMPORTANTE: los pie charts solo son útiles con POCAS categorías (2-5)
# Con más de 5 se vuelven ilegibles. En ML casi no se usan,
# pero para reportes ejecutivos son efectivos.
colores_region = [COLORES['norte'], COLORES['centro'], COLORES['sur']]
explode = (0.02, 0.05, 0.02)  # Separar ligeramente la región Centro

ax.pie(
    ventas_region.values,
    labels=ventas_region.index,
    autopct='%1.1f%%',          # Formato del porcentaje
    startangle=90,               # Empieza desde arriba (12 en punto)
    colors=colores_region,
    explode=explode,             # Destacar una porción
    shadow=True,
    textprops={'fontsize': 13, 'fontweight': 'bold'}
)
ax.set_title('🌍 Distribución de Ventas por Región', fontsize=15, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig('docs/09_pastel_region.png')
plt.close()
print("   ✅ Gráfico 9 guardado: docs/09_pastel_region.png")

# =====================================================================
# PASO 10: SUBPLOTS COMPLEJOS — Dashboard de ventas
# =====================================================================
print("\n" + "=" * 72)
print("🎛️ PASO 10: Dashboard — Múltiples gráficos en una figura")
print("=" * 72)
print("""
   📍 ¿Para qué sirve?
      Un solo vistazo a MÚLTIPLES dimensiones de los datos
      El dashboard final que resume todo el análisis
      Ideal para reportes ejecutivos o como portada de un análisis
""")

# ----------------------------------------------------------------------
# 10.1: Creamos una cuadrícula de 2×3 subplots
# ----------------------------------------------------------------------
# plt.subplots(2, 3) = 2 filas, 3 columnas = 6 gráficos
# figsize grande para que quepa todo
# squeeze=False para que axes sea siempre 2D
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
# axes ahora es una matriz de 2×3

# --- Dashboard: Ventas Diarias (línea) ---
# axes[0,0] = fila 0, columna 0
ventas_diarias_totales = df.groupby('fecha')['ventas'].sum()
axes[0, 0].plot(ventas_diarias_totales.index, ventas_diarias_totales.values,
                color='#2196F3', linewidth=1.5, alpha=0.7)
axes[0, 0].plot(tendencia_7d.index, tendencia_7d.values,
                color='#FF5722', linewidth=2.5, linestyle='--')
axes[0, 0].set_title('Ventas Diarias', fontsize=12, fontweight='bold')
axes[0, 0].tick_params(axis='x', rotation=30, labelsize=8)
axes[0, 0].grid(alpha=0.3, linestyle='--')

# --- Dashboard: Ventas por Categoría (barra) ---
ventas_cat_plot = df.groupby('categoria')['ventas'].sum().sort_values(ascending=False)
colores_bar = [COLORES[cat.lower().replace('ó', 'o').replace('é', 'e')]
               for cat in ventas_cat_plot.index]
axes[0, 1].bar(ventas_cat_plot.index, ventas_cat_plot.values, color=colores_bar,
               edgecolor='white', linewidth=1.5)
axes[0, 1].set_title('Ventas por Categoría', fontsize=12, fontweight='bold')
axes[0, 1].tick_params(axis='x', rotation=15, labelsize=9)
axes[0, 1].grid(axis='y', alpha=0.3, linestyle='--')

# --- Dashboard: Satisfacción (boxplot) ---
sns.boxplot(data=df, x='categoria', y='satisfaccion',
            palette=[COLORES['electronica'], COLORES['ropa'], COLORES['hogar']],
            ax=axes[0, 2], width=0.5)
axes[0, 2].set_title('Satisfacción por Categoría', fontsize=12, fontweight='bold')
axes[0, 2].grid(axis='y', alpha=0.3, linestyle='--')

# --- Dashboard: Marketing vs Ventas (scatter) ---
axes[1, 0].scatter(df['marketing'], df['ventas'], alpha=0.4, s=15, c='#9C27B0',
                   edgecolors='white', linewidth=0.3)
axes[1, 0].set_title('Marketing vs Ventas', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Marketing ($)', fontsize=9)
axes[1, 0].set_ylabel('Ventas', fontsize=9)
axes[1, 0].grid(alpha=0.3, linestyle='--')

# --- Dashboard: Distribución de ventas (histograma) ---
axes[1, 1].hist(df['ventas'], bins=30, color='#4CAF50', edgecolor='white', alpha=0.8)
axes[1, 1].axvline(df['ventas'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Media: {df["ventas"].mean():.0f}')
axes[1, 1].set_title('Distribución de Ventas', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Ventas', fontsize=9)
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(alpha=0.3, linestyle='--')

# --- Dashboard: Devoluciones vs Satisfacción (scatter) ---
sns.scatterplot(data=df, x='satisfaccion', y='devoluciones',
                hue='categoria', alpha=0.5, s=30,
                palette=[COLORES['electronica'], COLORES['ropa'], COLORES['hogar']],
                legend=False, ax=axes[1, 2])
axes[1, 2].set_title('Devoluciones vs Satisfacción', fontsize=12, fontweight='bold')
axes[1, 2].set_xlabel('Satisfacción', fontsize=9)
axes[1, 2].set_ylabel('Devoluciones', fontsize=9)
axes[1, 2].grid(alpha=0.3, linestyle='--')

plt.suptitle('🎛️ DASHBOARD COMPLETO — Análisis MarketPro',
             fontsize=18, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('docs/10_dashboard_completo.png', bbox_inches='tight')
plt.close()
print("   ✅ Gráfico 10 guardado: docs/10_dashboard_completo.png")

# =====================================================================
# 🧠 RESUMEN: Qué aprendiste hoy
# =====================================================================
print("\n" + "=" * 72)
print("🧠  RESUMEN: HOY APRENDISTE VISUALIZACIÓN DE DATOS")
print("=" * 72)
print("""
   📈 1. GRÁFICO DE LÍNEAS (plot)
        → Para ver tendencias en el tiempo
        → + rolling(window=7).mean() para suavizar ruido
    
   📊 2. GRÁFICO DE BARRAS (bar, barplot)
        → Para comparar categorías
        → sns.barplot() con hue para segunda dimensión
    
   📊 3. HISTOGRAMA (hist, histplot)
        → Para ver la distribución de una variable
        → + KDE para suavizar la curva
    
   📦 4. BOXPLOT (boxplot)
        → Para detectar outliers automáticamente
        → Anatomía: min, Q1, mediana, Q3, max, outliers
    
   🔵 5. SCATTER PLOT (scatter, scatterplot)
        → Para ver la relación entre dos variables
        → + hue para color por categoría
    
   🔥 6. HEATMAP (heatmap)
        → Para visualizar la matriz de correlación
        → Identificar qué features usar en ML

   🔗 7. PAIRPLOT (pairplot)
        → El rey del EDA: todas las relaciones en un solo gráfico
        → Lo que SIEMPRE haces primero con un dataset nuevo
    
   🥧 8. GRÁFICO DE PASTEL (pie)
        → Útil solo con 2-5 categorías
        → Para reportes ejecutivos, no para análisis técnico
    
   🎛️ 9. SUBPLOTS (subplots)
        → Dashboard de múltiples gráficos en una figura
        → Para resumir un análisis completo

   💡 PRINCIPIO CLAVE:
      Siempre visualiza tus datos ANTES de modelarlos.
      Un buen gráfico te dice en 1 segundo lo que 1000
      números no pueden expresar en 1 hora.
""")

print("   📂 Todos los gráficos están en: docs/")
print("   📊 Archivos generados:")
for f in sorted(os.listdir('docs')):
    if f.endswith('.png'):
        size = os.path.getsize(f'docs/{f}')
        print(f"      └─ {f} ({size//1024} KB)")
