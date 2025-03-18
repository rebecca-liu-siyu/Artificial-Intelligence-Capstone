import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, KMeansSMOTE
from imblearn.under_sampling import CondensedNearestNeighbour
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

    #class_weights = compute_class_weight('balanced', classes=np.unique(train_y), y=train_y.flatten())
    #class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    #print("Class Weights:", class_weight_dict)

    models = {
        "LR": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(max_depth=5)
    }

    drop_features = [0, 11]
    train_X_drop = np.delete(train_X, drop_features, axis=1)
    test_X_drop = np.delete(test_X, drop_features, axis=1)

    train_X_small, _, train_y_small, _ = train_test_split(train_X, train_y, train_size=0.25)

    smote = SMOTE()
    train_X_resampled, train_y_resampled = smote.fit_resample(train_X, train_y)

    '''
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(train_X)
    plt.pyplot.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=train_y.flatten(), palette='coolwarm', alpha=0.7)
    plt.pyplot.title("PCA Visualization of Original Classes")
    plt.pyplot.xlabel("Principal Component 1")
    plt.pyplot.ylabel("Principal Component 2")
    plt.pyplot.legend(title="Classes")
    plt.pyplot.show()
    '''

    for name, model in models.items():
        model.fit(train_X, train_y.flatten())
        predictions = model.predict(test_X)
        acc = accuracy_score(test_y, predictions)
        f1 = f1_score(test_y, predictions, average='macro')
        mcc = matthews_corrcoef(test_y, predictions)
        score = 0.3 * acc + 0.35 * f1 + 0.35 * mcc
        #print(confusion_matrix(test_y, predictions))
        print(f"---{name}---")
        print(f"{name} Accuracy: {acc:.4f}")
        print(f"{name} F1-Score: {f1:.4f}")
        print(f"{name} MCC: {mcc:.4f}")
        print(f"{name} Score: {score:.4f}")

if __name__ == "__main__":
    np.random.seed(0)
    main()