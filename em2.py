import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal, norm
import os
import atlas_utils as au
import em_helper as eh

np.seterr(all='ignore') 

#EM algorithm
class EM_algorithm:
    def __init__(self, n_components, method,mvm_method, max_iterations=50, tolerance=1e-5):
        #number of clusters
        self.n_components = n_components
        #iterations to stop the E-M loop. Either iterations or tolerance
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        #means, covariance and weights
        self.means = None
        self.covariances = None
        self.weights = None
        #methods to initialize: our multivariate gaussian or the one in scipy stats and parameters random or k-means
        self.method = method
        self.mvm_method = mvm_method

    def initialize_parameters_with_kmeans(self, data):
        #number of clusters is used to initialize kmeans
        kmeans = KMeans(n_clusters=self.n_components, n_init='auto')
        kmeans.fit(data)
        kmeans_labels = kmeans.labels_
        #function taken to make sure of the order of the clusters
        new_clusters = eh.robust_integration(kmeans.cluster_centers_,kmeans.predict(data))
        print(new_clusters)
        #assigning the clusters obtained with k-means and covariances as an identity matrix
        self.means = new_clusters
        self.covariances = [np.eye(data.shape[1]) for _ in range(self.n_components)]
        #weights are the proportion of the labels in the data
        self.weights = np.array([np.mean(kmeans_labels == i) for i in range(self.n_components)])

    def initialize_parameters_random(self, data):
        n_samples, n_features = data.shape
        #random number between 0 and 255, possible values in the grayscaleimage
        random_numbers = np.random.randint(0, 256, size=(self.n_components, 1))
        self.means = np.tile(random_numbers, (1, n_features))
        self.covariances = [np.eye(data.shape[1]) for _ in range(self.n_components)]
        self.weights = np.full(self.n_components, 1.0 / self.n_components)
        new_clusters = eh.robust_integration(self.means,self.expectation(data))
        #in the class material, it is mentioned that parameters can be initalized either with a "set of parameters and then expectation"
        #or the weights and maximization steps, that is way we use it in the function to reorganize clusters
        self.means = new_clusters
        print(self.means)

    def initialize_parameters_intensity_information(self, data, em_params):
        # Set means based on em_params
        self.means = np.array([[em_params['csf_model']['mean']], 
                                  [em_params['gm_model']['mean']], 
                                  [em_params['wm_model']['mean']]])

        # Set covariances based on em_params
        self.covariances = [np.array([[em_params['csf_model']['sd'] ** 2]]), 
                                np.array([[em_params['gm_model']['sd'] ]]), 
                                np.array([[em_params['wm_model']['sd'] ]])]
        self.weights = np.full(self.n_components, 1.0 / self.n_components)

        for n in range(self.n_components):
          print(self.means[n])
          print(self.covariances[n])

    def initialize_parameters(self, data, em_params):
        #initialize according to the parameter initialization method selected
        if self.method == 'km':
          return self.initialize_parameters_with_kmeans(data)
        if self.method == 'rn':
          return self.initialize_parameters_random(data)
        if self.method == 'ii':
          return self.initialize_parameters_intensity_information(data, em_params)

    def expectation(self, data, belonging):
        # Create an array to store the 'belonging' scores
        #belonging = np.zeros((data.shape[0], self.n_components))
        
        for k in range(self.n_components):
            if self.mvm_method == 'mvn':
                # Use scipy.stats.multivariate_normal
                mvn = multivariate_normal(mean=self.means[k], cov=self.covariances[k], allow_singular=True)
                belonging[:, k] = self.weights[k] * mvn.pdf(data)
            elif self.mvm_method == 'our':
                # Use our multivariate gaussian function
                belonging[:, k] = self.weights[k] * eh.multivariate_normal_nonp(self.means[k], self.covariances[k], data)

        # Normalize the probabilities so they sum to 1 across the tissue types
        normalization = belonging.sum(axis=1, keepdims=True) + 1e-6
        belonging /= normalization
        return belonging

    def maximization(self, data, belonging):
        n_samples = data.shape[0]

        for k in range(self.n_components):
            # Weighted sum for each cluster
            weighted_sum = np.dot(belonging[:, k], data)

            # Update the mean for each cluster
            self.means[k] = weighted_sum / belonging[:, k].sum()

            # Update covariance matrices
            diff = data - self.means[k]
            weighted_diff = belonging[:, k][:, np.newaxis] * diff
            self.covariances[k] = np.dot(weighted_diff.T, diff) / belonging[:, k].sum()

        # Update weights
        self.weights = belonging.sum(axis=0) / n_samples

    def fit(self, data, belonging_, em_params, base_name):
        self.initialize_parameters(data, em_params)

        likelihoods = []  # List to store likelihood values per iteration
        prev_log_likelihood = 0  # Initialize previous log likelihood
        belonging = belonging_

        # Iterate EM algorithm for a maximum number of iterations
        for iteration in range(self.max_iterations):
            belonging = self.expectation(data, belonging_)
            self.maximization(data, belonging)
            
            # Efficient computation of log likelihood
            log_likelihood = np.sum(np.log(belonging.sum(axis=1) + 1e-6))  # Add a small value to prevent log(0)
            likelihoods.append(log_likelihood)  # Append the likelihood value
            
            # Check for convergence
            if iteration > 0 and abs(log_likelihood - prev_log_likelihood) < self.tolerance:
                break
            prev_log_likelihood = log_likelihood

        # Printing final clusters
        print(self.means)

        # Plotting likelihood convergence
        plt.figure()
        plt.plot(range(1, len(likelihoods) + 1), likelihoods)
        plt.xlabel('Iteration')
        plt.ylabel('Log Likelihood')
        plt.title('Likelihood per Iteration')
        plt.savefig(f"{base_name}_em2_likelihood.png",bbox_inches='tight')
        plt.close()

    def predict(self, data, belonging_):
        belonging = self.expectation(data, belonging_)
        return np.argmax(belonging, axis=1) + 1 #need to add 1 since labels at this point are 0,1,2 and groundtruth is 1,2,3

def process_images_with_em(testing_folder, em_params):
    images_folder = os.path.join(testing_folder, 'testing-images')
    labels_folder = os.path.join(testing_folder, 'testing-labels')
    masks_folder = os.path.join(testing_folder, 'testing-mask')
    dice_scores = {}

    for image_file in os.listdir(images_folder):
        base_name = image_file.split('.')[0]
        image_path = os.path.join(images_folder, image_file)
        label_path = os.path.join(labels_folder, f"{base_name}_3C.nii.gz")
        mask_path = os.path.join(masks_folder, f"{base_name}_1C.nii.gz")

        image_data = nib.load(image_path).get_fdata()
        label_data = nib.load(label_path).get_fdata()
        mask_data = nib.load(mask_path).get_fdata()

        masked_image_data = au.apply_mask(image_data, mask_data)
        old_min, old_max = np.min(masked_image_data), np.max(masked_image_data)
        normalized_image_data = au.normalize_image(masked_image_data, old_min, old_max)

        tissue_map, normalized_probs = au.segment_tissue_with_intensity(normalized_image_data, 
                                                                     em_params['csf_model'], 
                                                                     em_params['wm_model'], 
                                                                     em_params['gm_model'], 
                                                                     nib.load(image_path).affine, 
                                                                     "test.nii.gz", mask_data)
        relevant_data, nonzero_indices = eh.create_data_nonzero(normalized_image_data)
        # Initialize and run EM algorithm
        aligned = eh.align_probs_with_nonzero_data(normalized_probs, nonzero_indices)
        print(aligned.shape)
        em = EM_algorithm(n_components=3, method='ii', mvm_method='mvn', max_iterations=15, tolerance=1e-5)
        #em.weights = normalized_probs.reshape(-1, 3).mean(axis=0)  # Initialize weights with normalized probabilities
        em.fit(relevant_data, aligned, em_params, base_name)
        em_tissue_map = em.predict(relevant_data, aligned)

        final_array = eh.final_array_calculator(normalized_image_data, nonzero_indices, em_tissue_map,mask_data)

        # Calculate Dice scores
        dice_csf = au.dice_score_per_tissue(label_data, final_array, 1)
        dice_wm = au.dice_score_per_tissue(label_data, final_array, 2)
        dice_gm = au.dice_score_per_tissue(label_data, final_array, 3)
        dice_scores[base_name] = {'EM_CSF': dice_csf, 'EM_WM': dice_wm, 'EM_GM': dice_gm}

        print(f"EM Dice/Jaccard score for {base_name}: CSF - {dice_csf}, WM - {dice_wm}, GM - {dice_gm}")
        au.plot_comparison(final_array, label_data, title1="EM Segmented Tissue", title2="Ground Truth", 
                        slice_nums=(136, 142, 163), save_filename=f"{base_name}_em2kmeans.png")

    return dice_scores