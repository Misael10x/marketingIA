from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import plotly
import tensorflow as tf
import os

app = Flask(__name__)

# limitar tamaño de archivos (5MB)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024


# -----------------------------
# cargar modelo tflite
# -----------------------------

model_path = "autoencoder.tflite"

interpreter = None
input_details = None
output_details = None

if os.path.exists(model_path):

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Modelo TFLite cargado")

else:
    print("Modelo no encontrado")


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

    file = request.files["file"]

    if file.filename == "":
        return "No file uploaded"

    df = pd.read_csv(file, encoding="latin1")

    # limitar dataset para evitar crash en render
    df = df.head(500)

    # solo columnas numericas
    df = df.select_dtypes(include=[np.number])

    if df.shape[1] == 0:
        return "CSV must contain numeric columns"

    data = df.values.astype("float32")

    # -------------------------
    # HEATMAP
    # -------------------------

    heatmap = px.imshow(df.corr(), title="Correlation Heatmap")
    heatmap_html = plotly.io.to_html(heatmap, full_html=False)

    # -------------------------
    # HISTOGRAMA
    # -------------------------

    hist = px.histogram(df, x=df.columns[0])
    hist_html = plotly.io.to_html(hist, full_html=False)

    # -------------------------
    # BOXPLOT
    # -------------------------

    box = px.box(df)
    box_html = plotly.io.to_html(box, full_html=False)

    # -------------------------
    # SCATTER MATRIX
    # -------------------------

    if len(df.columns) > 6:
        scatter = px.scatter_matrix(df[df.columns[:6]])
    else:
        scatter = px.scatter_matrix(df)

    scatter_html = plotly.io.to_html(scatter, full_html=False)

    # -------------------------
    # CLUSTER
    # -------------------------

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

    # -------------------------
    # PCA 3D
    # -------------------------

    if df.shape[1] >= 3:

        pca3 = PCA(n_components=3)

        pca_data3 = pca3.fit_transform(df)

        pca3_df = pd.DataFrame({
            "x": pca_data3[:,0],
            "y": pca_data3[:,1],
            "z": pca_data3[:,2]
        })

        pca3_plot = px.scatter_3d(
            pca3_df,
            x="x",
            y="y",
            z="z",
            title="PCA 3D Visualization"
        )

        pca3_html = plotly.io.to_html(pca3_plot, full_html=False)

    else:
        pca3_html = "<p>Not enough columns for PCA 3D</p>"

    # -------------------------
    # FEATURE IMPORTANCE
    # -------------------------

    importance = df.var().sort_values(ascending=False)

    importance_df = pd.DataFrame({
        "feature": importance.index,
        "importance": importance.values
    })

    importance_plot = px.bar(
        importance_df,
        x="feature",
        y="importance",
        title="Feature Importance"
    )

    importance_html = plotly.io.to_html(importance_plot, full_html=False)

    # -------------------------
    # OUTLIERS
    # -------------------------

    z_scores = np.abs((df - df.mean()) / df.std())

    outliers = (z_scores > 3).sum(axis=1)

    outlier_df = pd.DataFrame({
        "index": range(len(outliers)),
        "outliers": outliers
    })

    outlier_plot = px.scatter(
        outlier_df,
        x="index",
        y="outliers",
        title="Outlier Detection"
    )

    outlier_html = plotly.io.to_html(outlier_plot, full_html=False)

    # -------------------------
    # TREND
    # -------------------------

    trend_plot = px.line(
        df.reset_index(),
        x="index",
        y=df.columns[0],
        title="Trend Analysis"
    )

    trend_html = plotly.io.to_html(trend_plot, full_html=False)

    # -------------------------
    # AUTOENCODER
    # -------------------------

    error_html = None

    if interpreter is not None:

        errors = []

        for row in data:

            row = list(row)

            if len(row) < 37:
                row = row + [0]*(37-len(row))

            values = np.array(row, dtype=np.float32).reshape(1,37)

            interpreter.set_tensor(input_details[0]['index'], values)

            interpreter.invoke()

            prediction = interpreter.get_tensor(output_details[0]['index'])

            error = np.mean(np.square(values - prediction))

            errors.append(error)

        error_df = pd.DataFrame({
            "index": range(len(errors)),
            "error": errors
        })

        error_plot = px.line(
            error_df,
            x="index",
            y="error",
            title="Anomaly Detection"
        )

        error_html = plotly.io.to_html(error_plot, full_html=False)

    # -------------------------
    # AI INSIGHTS
    # -------------------------

    rows = df.shape[0]
    cols = df.shape[1]

    insights = f"""
    Rows: {rows}<br>
    Columns: {cols}<br>
    Highest variance feature: {importance.index[0]}<br>
    Mean of first column: {df[df.columns[0]].mean():.2f}
    """

    # -------------------------
    # DASHBOARD
    # -------------------------

    return render_template(
        "dashboard.html",
        heatmap=heatmap_html,
        hist=hist_html,
        box=box_html,
        scatter=scatter_html,
        cluster=cluster_html,
        pca3=pca3_html,
        importance=importance_html,
        outliers=outlier_html,
        trend=trend_html,
        error=error_html,
        insights=insights
    )


# -----------------------------
# RUN SERVER
# -----------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(host="0.0.0.0", port=port)