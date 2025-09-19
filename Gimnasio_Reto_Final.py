# -*- coding: utf-8 -*-
# Análisis de datos + K-Means (sin gráficas de K-Means)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# 1) Cargar datos con manejo de errores
try:
    df = pd.read_csv("escenario_gimnasios_15.csv", encoding="utf-8-sig")
except FileNotFoundError:
    print("❌ No se encontró el archivo CSV")
    exit()

print("Forma (filas, columnas):", df.shape)
print("\nTipos de datos:\n", df.dtypes)
print("\nValores faltantes por columna:\n", df.isna().sum())

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

desc = df.describe(include="all")
print("\nDescripción general:\n", desc)

# 1.5) Medidas de tendencia central
var1 = "Ingresos_Mensuales"
var2 = "Precio_Membresia"

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

# 2) Funciones para gráficos iniciales
def graficos_estaticos(df, var1, var2, corr):
    # Boxplot Ingresos
    plt.figure(figsize=(6, 4))
    df[var1].plot(kind="box", title=f"Boxplot - {var1}")
    plt.tight_layout()
    plt.savefig("boxplot_ingresos.png", dpi=120)
    plt.close()

    # Histograma Ingresos
    plt.figure(figsize=(6, 4))
    df[var1].plot(kind="hist", bins=10, title=f"Histograma - {var1}")
    plt.xlabel("Valor")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig("hist_ingresos.png", dpi=120)
    plt.close()

    # Boxplot Precio Membresía
    plt.figure(figsize=(6, 4))
    df[var2].plot(kind="box", title=f"Boxplot - {var2}")
    plt.tight_layout()
    plt.savefig("boxplot_precio_membresia.png", dpi=120)
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

# 3) Preparación para clustering con K-Means
num = df.select_dtypes("number").copy()
num = num.fillna(num.mean(numeric_only=True))

# Eliminar variables muy correlacionadas
high_corr_threshold = 0.90
corr_matrix = num.corr()
to_drop = set()
cols = corr_matrix.columns.tolist()
for i, c1 in enumerate(cols):
    for c2 in cols[i+1:]:
        r = corr_matrix.loc[c1, c2]
        if pd.notna(r) and abs(r) >= high_corr_threshold:
            to_drop.add(c2)

X = num.drop(columns=list(to_drop)) if to_drop else num.copy()
print("\n=== Selección de variables para clustering ===")
print("Variables usadas:", list(X.columns))
if to_drop:
    print("Eliminadas por alta correlación (|r|>=0.90):", list(to_drop))

# Escalado
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# Método del codo y silueta
k_values = range(2, 7)
inertias = []
sil_scores = []
for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(Xs)
    inertias.append(km.inertia_)
    sil = silhouette_score(Xs, labels)
    sil_scores.append(sil)

best_k = int(k_values[np.argmax(sil_scores)])
print(f"\nSugerencia automática de k (por máxima silueta en 2..6): k={best_k}")

# Entrenar K-Means con el k óptimo
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(Xs)

# Centros en escala original
centers_std = kmeans.cluster_centers_
centers_orig = scaler.inverse_transform(centers_std)
centers_df = pd.DataFrame(centers_orig, columns=X.columns)
centers_df.index = [f"Cluster_{i}" for i in range(best_k)]
print("\n📌 Centros de K-Means (en escala original):")
print(centers_df.round(2))

# Conteo de observaciones por cluster
print("\n📊 Conteo de observaciones por cluster:")
print(df["Cluster"].value_counts())

# Medias por cluster
print("\n📊 Medias por cluster:")
print(df.groupby("Cluster")[variables_clave].mean())

# 4) Menú de selección de gráficos
print("\n📊 Selecciona qué tipo de gráficos quieres ver:")
print("1 - Solo gráficos estáticos (PNG)")
print("2 - Solo gráficos interactivos (Plotly)")
print("3 - Ambos")

opcion = input("Elige una opción (1, 2 o 3): ")

corr = num.corr()
if opcion == "1":
    graficos_estaticos(df, var1, var2, corr)
elif opcion == "2":
    graficos_interactivos(df, var1, var2, corr)
elif opcion == "3":
    graficos_estaticos(df, var1, var2, corr)
    graficos_interactivos(df, var1, var2, corr)
else:
    print("❌ Opción no válida. Intenta de nuevo.")

print("\n✅ Análisis completado.")