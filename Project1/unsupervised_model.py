import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from preprocessor import Preprocessor

def dataPreprocessing():
    train_X_raw = pd.read_csv('dataset/train_X.csv')
    train_y_raw = pd.read_csv('dataset/train_y.csv')
    test_X_raw = pd.read_csv('dataset/test_X.csv')
    test_y_raw = pd.read_csv('dataset/test_y.csv')

    preprocessor_train = Preprocessor(train_X_raw)
    train_X = preprocessor_train.preprocess()

    preprocessor_test = Preprocessor(test_X_raw)
    test_X = preprocessor_test.preprocess()

    train_X = np.array(train_X)
    train_y = np.array(train_y_raw)
    test_X = np.array(test_X)
    test_y = np.array(test_y_raw)

    return train_X, train_y, test_X, test_y

def main():
    train_X, train_y, test_X, test_y = dataPreprocessing()

    print("data preprocessing is successful")

    pca = PCA(n_components=10)
    train_X_pca = pca.fit_transform(train_X)

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(train_X_pca)

    labels = kmeans.labels_
    silhouette_avg = silhouette_score(train_X_pca, labels)
    print(f"Silhoutte Score: {silhouette_avg}")
    train_y = train_y - 1
    ari_score = adjusted_rand_score(train_y.flatten(), labels)
    print(f"Adjusted Rand Index: {ari_score}")

    clusters = kmeans.predict(train_X_pca)

    df = pd.DataFrame({"True_Label": train_y.flatten(), "Cluster_Label": clusters})
    print(df.groupby("Cluster_Label")["True_Label"].value_counts())

    print("Cluster centers:", kmeans.cluster_centers_)
    print("Labels:", clusters)

if __name__ == "__main__":
    np.random.seed(0)
    main()