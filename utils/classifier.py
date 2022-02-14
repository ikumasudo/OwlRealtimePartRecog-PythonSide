import numpy as np
import pickle


class Classifier:
    def __init__(self, path: str):
        with open(path, "rb") as f:
            self.model = pickle.load(f)
            
    def predict(self, data: np.ndarray) -> np.ndarray:
        return self.model.predict(data)
    
    def predict_proba(self, data: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(data)
    
if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore")
    
    from preprocess import Preprocess
    import pandas as pd
    
    data = pd.read_csv("./data/sample.csv", header=None).values.transpose()[1:]
    prep = Preprocess("./data/OwlNotebook22-FeatureImportance.csv", 100)
    data = prep(data)
    print(data.shape, type(data))
    
    clf = Classifier("./data/OwlNotebook22-LR-100.pickle")
    print(clf.model.classes_)
    print(all(clf.model.feature_names_in_ == data.columns.tolist()))
    
    pred = clf.predict(data)
    print(pred)
    # print(clf.predict_proba(data))