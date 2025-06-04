import os
import pickle
import numpy as np
from sklearn.cluster import KMeans

class Predictor:
    """Predictor that mimics the training logic from the notebook."""

    def __init__(self):
        self.model = None

    def prepPredictor(self):
        """Load existing model or train a simple one if missing."""
        model_path = os.path.join(os.path.dirname(__file__), 'trained_Model.pkl')
        if os.path.isfile(model_path):
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
        else:
            # generate synthetic training data similar to the notebook structure
            np.random.seed(42)
            data = np.random.randint(1, 6, size=(100, 50))
            self.model = KMeans(n_clusters=5, random_state=42)
            self.model.fit(data)
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)

    def predict(self, data):
        if self.model is None:
            raise ValueError('Model has not been prepared.')
        prediction = self.model.predict(np.array(data))
        return prediction[0]
