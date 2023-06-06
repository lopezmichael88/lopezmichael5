import numpy as np
from sklearn.mixture import GaussianMixture

class GaussianMixtureModel:
    def __init__(self, num_components):
        self.num_components = num_components
        self.gmm = GaussianMixture(n_components=num_components)

    def fit(self, data):
        self.gmm.fit(data)

    def generate_samples(self, num_samples):
        return self.gmm.sample(num_samples)

    def calculate_log_likelihood(self, data):
        return self.gmm.score_samples(data)

if __name__ == '__main__':
    # Example usage: Fit a Gaussian Mixture Model (GMM) to a dataset

    # Generate a synthetic dataset
    np.random.seed(0)
    num_samples = 1000
    data = np.concatenate([np.random.normal(loc=0, scale=1, size=(num_samples // 2, 2)),
                           np.random.normal(loc=5, scale=1, size=(num_samples // 2, 2))])

    # Fit a GMM to the dataset
    num_components = 2
    gmm_model = GaussianMixtureModel(num_components)
    gmm_model.fit(data)

    # Generate samples from the trained GMM
    num_generated_samples = 100
    generated_samples, _ = gmm_model.generate_samples(num_generated_samples)

    # Calculate the log-likelihood of the original data
    log_likelihood = gmm_model.calculate_log_likelihood(data)

    print("Generated samples:", generated_samples)
    print("Log-likelihood of the data:", log_likelihood)
