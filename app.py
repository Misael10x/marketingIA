from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import plotly
import os

app = Flask(__name__)

# -----------------------------
# HOME
# -----------------------------

@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# ANALISIS CSV
# -----------------------------

@app.route("/upload_csv", methods=["POST"])
def upload_csv():

    try:

        if "file" not in request.files:
            return "No se recibió archivo"

        file = request.files["file"]

        if file.filename == "":
            return "Archivo vacío"

        print("Archivo recibido:", file.filename)

        # leer csv
        df = pd.read_csv(file, encoding="latin1")

        # solo columnas numericas
        df = df.select_dtypes(include=[np.number])

        if df.shape[1] == 0:
            return "El CSV no tiene columnas numéricas"

        # limpiar datos
        df = df.fillna(0)

        # -----------------------------
        # LIMITES PARA EVITAR 502
        # -----------------------------

        if df.shape[1] > 10:
            df = df.iloc[:, :10]

        if df.shape[0] > 1000:
            df = df.sample(1000)

        # -----------------------------
        # HEATMAP
        # -----------------------------

        heatmap = px.imshow(df.corr(), title="Correlation Heatmap")

        heatmap_html = plotly.io.to_html(heatmap, full_html=False)

        # -----------------------------
        # HISTOGRAMA
        # -----------------------------

        hist = px.histogram(df, x=df.columns[0])

        hist_html = plotly.io.to_html(hist, full_html=False)

        # -----------------------------
        # BOXPLOT
        # -----------------------------

        box = px.box(df)

        box_html = plotly.io.to_html(box, full_html=False)

        # -----------------------------
        # SCATTER MATRIX
        # -----------------------------

        scatter = px.scatter_matrix(df)

        scatter_html = plotly.io.to_html(scatter, full_html=False)

        # -----------------------------
        # CLUSTER
        # -----------------------------

        kmeans = KMeans(n_clusters=3, n_init=10)

        clusters = kmeans.fit_predict(df)

        pca = PCA(n_components=2)

        pca_data = pca.fit_transform(df)

        cluster_df = pd.DataFrame({
            "PCA1": pca_data[:,0],
            "PCA2": pca_data[:,1],
            "cluster": clusters
        })

        cluster_plot = px.scatter(
            cluster_df,
            x="PCA1",
            y="PCA2",
            color="cluster",
            title="Customer Clusters"
        )

        cluster_html = plotly.io.to_html(cluster_plot, full_html=False)

        # -----------------------------
        # PCA 3D
        # -----------------------------

        components = min(3, df.shape[1])

        pca3 = PCA(n_components=components)

        pca_data3 = pca3.fit_transform(df)

        pca3_df = pd.DataFrame({
            "x": pca_data3[:,0],
            "y": pca_data3[:,1],
            "z": pca_data3[:,2] if components == 3 else np.zeros(len(pca_data3))
        })

        pca3_plot = px.scatter_3d(
            pca3_df,
            x="x",
            y="y",
            z="z",
            title="PCA 3D Visualization"
        )

        pca3_html = plotly.io.to_html(pca3_plot, full_html=False)

        return render_template(
            "dashboard.html",
            heatmap=heatmap_html,
            hist=hist_html,
            box=box_html,
            scatter=scatter_html,
            cluster=cluster_html,
            pca3=pca3_html,
            error=None
        )

    except Exception as e:

        return f"Error procesando CSV: {str(e)}"


# -----------------------------
# RUN SERVER
# -----------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(host="0.0.0.0", port=port)