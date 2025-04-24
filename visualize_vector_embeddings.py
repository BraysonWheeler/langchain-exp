from sklearn.decomposition import PCA
import plotly.express as px


def visualize_vector_embeddings(vector: list[list[float]]):
    pca = PCA(n_components=2)  # 2D
    data = pca.fit_transform(vector)
    scatter = px.scatter(data, data[:, 0], data[:, 1])
    scatter.show()
