import numpy as np
import pandas as pd
import atlas_utils as au

#####
#here some methods of the EM class that were repeated among em files. Could have been refined.

#this method was found in fitushar repo and added because the centers used to change every time kmeans was executed
def robust_integration(centroids, Kmeans_predict):
        # Find the minimum and maximum values and their corresponding indexes
        min_index = np.argmin(centroids[:, 0])
        max_index = np.argmax(centroids[:, 0])

        # Make clustering more robust
        centroid_new = np.zeros_like(centroids)
        centroid_new[0] = centroids[min_index]
        centroid_new[2] = centroids[max_index]

        if min_index + max_index == 1:
            centroid_new[1] = centroids[2]
        elif min_index + max_index == 2:
            centroid_new[1] = centroids[1]
        elif min_index + max_index == 3:
            centroid_new[1] = centroids[0]
        return centroid_new

def multivariate_normal_nonp(mean, cov, x):
        """
        Calculate the probability density of a multivariate normal distribution at point x.

        Parameters, same as scipy stats multivariate_normal:
        - mean: Mean vector of the distribution.
        - cov: Covariance matrix of the distribution.
        - x: Data point at which to calculate the probability density.

        Returns:
        - pdf: Probability density at point x.
        """
        dim = mean.shape[0]

        # Calculate the determinant of the covariance matrix for the denominator
        det = np.linalg.det(cov)
        # Add regularization term (small constant) to the diagonal of the covariance matrix
        reg_param = 1e-6
        reg_cov = cov + reg_param * np.eye(cov.shape[0])

        # Calculate the inverse of the regularized covariance matrix
        inv_cov = np.linalg.inv(reg_cov)
        # Centered data point
        x_minus_mean = x - mean

        # Exponent term
        exponent = -0.5 * np.sum(np.dot(x_minus_mean, inv_cov) * x_minus_mean, axis=1) #mahalanobis distance

        # Normalization term
        normalization = 1.0 / (((2 * np.pi) ** (dim / 2)) * np.sqrt(det))

        # Probability density
        pdf = normalization * np.exp(exponent)

        return pdf

def create_data_nonzero(*nifti_images):
    feature_data = []
    #done this way to accept more than one image
    #mascara is to get rid of the cerebellum
    for image in nifti_images:
        #nifti_data = nib.load(image).get_fdata()
        #nifti_data = np.multiply(nifti_data,mascara)
        flattened_data = image.flatten()
        feature_data.append(flattened_data)

    # Stack all the data together
    feature_data = np.vstack(feature_data)
    feature_data = np.transpose(feature_data)

    # Find row indices with nonzero data
    feature_data_nonzero_row_indices = [i for i, x in enumerate(feature_data) if x.any()]
    data_nonzero = feature_data[feature_data_nonzero_row_indices]
    print(data_nonzero.shape)
    return data_nonzero, feature_data_nonzero_row_indices

def align_probs_with_nonzero_data(normalized_probs, nonzero_indices):
    # Flatten normalized_probs while preserving the tissue type dimension
    flat_probs = normalized_probs.reshape(normalized_probs.shape[0], -1)  # shape becomes (3, x*y*z)

    # Filter out the probabilities corresponding to nonzero data points
    aligned_probs = flat_probs[:, nonzero_indices]

    # Transpose to match shape (n_samples, n_components)
    aligned_probs = aligned_probs.T  # shape becomes (n_samples, 3)

    return aligned_probs

def export_dice_scores(testing_folder, dice_scores,filename):
    data = []
    for image_number, scores in dice_scores.items():
        row = {'Image': image_number,'EM_CSF': scores['EM_CSF'],'EM_WM': scores['EM_WM'],'EM_GM': scores['EM_GM']}
        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(df)

def calculate_tissue_intensity_stats(normalized_image_data, tissue_map):
    tissue_stats = {}

    # For each tissue type (1, 2, 3)
    for tissue_type in [1, 2, 3]:
        # Mask the normalized data with the tissue map
        tissue_masked = normalized_image_data[tissue_map == tissue_type]

        # Calculate mean and standard deviation
        mean = np.mean(tissue_masked)
        sd = np.std(tissue_masked)

        # Store in the dictionary
        tissue_stats[f'tissue_{tissue_type}'] = {'mean': mean, 'sd': sd}

    return tissue_stats

def final_array_calculator(normalized_image_data, nonzero_indices, em_tissue_map,mask_data):
    shape_original_image = normalized_image_data.shape
    shape_original_image_flatten = normalized_image_data.flatten().shape
    segmented_image=np.zeros(shape_original_image_flatten)
    segmented_image[nonzero_indices]=em_tissue_map
    segmented=np.reshape(segmented_image,shape_original_image)

    modified_array = np.where(segmented == 3, 2, segmented)
    multiplied_array = modified_array * segmented
    final_array = np.where(multiplied_array == 4, 3, np.where(multiplied_array == 6, 2, multiplied_array))
    final_array = au.apply_mask(final_array, mask_data)

    return final_array

def calculate_weights_from_tissue_maps(tissue_map):
    csf_count = np.sum(tissue_map == 1)
    gm_count = np.sum(tissue_map == 3)
    wm_count = np.sum(tissue_map == 2)

    total = csf_count + gm_count + wm_count
    if total == 0:
        return np.full(3, 1/3)

    return np.array([csf_count, gm_count, wm_count]) / total