import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff

# -------------------------------------------------------------
# 1) Cargar datos con manejo de errores
# -------------------------------------------------------------
try:
    df = pd.read_csv("escenario_tiendas_15.csv", encoding="utf-8-sig")
except FileNotFoundError:
    print("❌ No se encontró el archivo CSV")
    exit()

print("Forma (filas, columnas):", df.shape)
print("\nTipos de datos:\n", df.dtypes)
print("\nValores faltantes por columna:\n", df.isna().sum())

# Mostrar todas las columnas y ancho amplio
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

desc = df.describe(include="all")
print(desc)

# Estadística básica
desc = df.describe(include="all")
print("\nDescripción numérica:\n", desc)

# Variables clave
var1 = "Ventas_Mensuales"
var2 = "Precio_Promedio"

# Matriz de correlación
num = df.select_dtypes("number")
corr = num.corr()
print("\nMatriz de correlación:\n", corr)

# -------------------------------------------------------------
# 1.5) Medidas de tendencia central
# -------------------------------------------------------------
def mostrar_medidas(df, variables):
    print("\n📌 Medidas de tendencia central")
    print("-" * 40)
    for var in variables:
        if var in df.columns:
            media = df[var].mean(skipna=True)
            mediana = df[var].median(skipna=True)
            moda = df[var].mode(dropna=True)
            moda = moda.iloc[0] if not moda.empty else None
            print(f"{var}:")
            print(f"  Media   = {media:.2f}")
            print(f"  Mediana = {mediana:.2f}")
            print(f"  Moda    = {moda}")
            print("-" * 20)

variables_clave = [var1, var2]
mostrar_medidas(df, variables_clave)

# -------------------------------------------------------------
# 2) Funciones para gráficos
# -------------------------------------------------------------
def graficos_estaticos(df, var1, var2, corr):
    # Boxplot Ventas
    plt.figure(figsize=(6,4))
    df[var1].plot(kind="box", title=f"Boxplot - {var1}")
    plt.tight_layout()
    plt.savefig("boxplot_ventas.png", dpi=120)
    plt.close()

    # Histograma Ventas
    plt.figure(figsize=(6,4))
    df[var1].plot(kind="hist", bins=10, title=f"Histograma - {var1}")
    plt.xlabel("Valor")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig("hist_ventas.png", dpi=120)
    plt.close()

    # Boxplot Precio Promedio
    plt.figure(figsize=(6,4))
    df[var2].plot(kind="box", title=f"Boxplot - {var2}")
    plt.tight_layout()
    plt.savefig("boxplot_precio_promedio.png", dpi=120)
    plt.close()

    # Heatmap de correlaciones
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=0.8)
    sns.heatmap(corr, annot=True, fmt=".2f", square=True,
                annot_kws={"size": 7}, cmap="coolwarm", cbar_kws={"shrink": 0.8})
    plt.title("Heatmap de correlaciones (variables numéricas)", fontsize=10)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig("heatmap_correlaciones.png", dpi=120)
    plt.close()

    print("\n✅ Gráficos estáticos guardados como imágenes PNG.")

def graficos_interactivos(df, var1, var2, corr):
    # Boxplot interactivo
    fig_box = px.box(df, y=var1, title=f"Boxplot interactivo - {var1}")
    fig_box.show()

    # Histograma interactivo
    fig_hist = px.histogram(df, x=var1, nbins=15, title=f"Histograma interactivo - {var1}")
    fig_hist.show()

    # Scatter interactivo
    fig_scatter = px.scatter(df, x=var2, y=var1,
                             size=var1, color=var2,
                             hover_data=df.columns,
                             title=f"Relación entre {var1} y {var2}")
    fig_scatter.show()

    # Heatmap interactivo
    fig_heatmap = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        annotation_text=corr.round(2).values,
        colorscale="RdBu",
        showscale=True
    )
    fig_heatmap.update_layout(title="Heatmap interactivo de correlaciones")
    fig_heatmap.show()

    print("\n✅ Gráficos interactivos desplegados en el navegador/ventana.")

# -------------------------------------------------------------
# 3) Menú de selección de gráficos
# -------------------------------------------------------------
print("\n📊 Selecciona qué tipo de gráficos quieres ver:")
print("1 - Solo gráficos estáticos (PNG)")
print("2 - Solo gráficos interactivos (Plotly)")
print("3 - Ambos")

opcion = input("Elige una opción (1, 2 o 3): ")

if opcion == "1":
    graficos_estaticos(df, var1, var2, corr)
elif opcion == "2":
    graficos_interactivos(df, var1, var2, corr)
elif opcion == "3":
    graficos_estaticos(df, var1, var2, corr)
    graficos_interactivos(df, var1, var2, corr)
else:
    print("❌ Opción no válida. Intenta de nuevo.")

# -------------------------------------------------------------
# 4) Correlaciones más fuertes (sin incluir la propia variable)
# -------------------------------------------------------------
if var1 in corr.columns:
    top_corr = corr[var1].drop(labels=[var1]).abs().sort_values(ascending=False).head(5)
    print(f"\n🔥 Correlaciones más fuertes con {var1}:\n", top_corr)

print("\n✅ Análisis completado.")