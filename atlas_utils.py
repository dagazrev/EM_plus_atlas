import nibabel as nib
import numpy as np
import os
import itk
import shutil
import dataloader as dl
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma
import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches


def split_nii_channels(input_file_path):
    img = nib.load(input_file_path)
    data = img.get_fdata()

    if data.shape[-1] != 4:
        raise ValueError("The input image does not have 4 channels")

    for i in range(4):
        channel_data = data[..., i]
        channel_img = nib.Nifti1Image(channel_data, img.affine, img.header)

        output_file_path = os.path.splitext(input_file_path)[0] + f"_channel_{i}.nii.gz"
        nib.save(channel_img, output_file_path)
        print(f"Channel {i} saved to {output_file_path}")


def convert_nii_to_niigz(input_file_path):
#to reduce size of the prob maps since in part A those were heavy-sized files
    img = nib.load(input_file_path)
    output_file_path = os.path.splitext(input_file_path)[0] + ".nii.gz"
    nib.save(img, output_file_path)
    print(f"File saved as {output_file_path}")

def register_image(data_loader, parameter_files, atlases_folder):
    prob_csf_path = os.path.join(atlases_folder, 'mni_CSF.nii.gz')
    prob_wm_path = os.path.join(atlases_folder, 'mni_WM.nii.gz')
    prob_gm_path = os.path.join(atlases_folder, 'mni_GM.nii.gz')
    moving_image_path = os.path.join(atlases_folder, 'template.nii.gz')

    image_files, _, mask_files = data_loader.find_files()
    for image_file, mask_file in zip(image_files, mask_files):
        fixed_image_path = os.path.join(data_loader.images_folder, image_file)
        print(fixed_image_path)
        fixed_mask_path = os.path.join(data_loader.masks_folder, mask_file)

        output_folder = os.path.join(data_loader.data_folder, image_file.split('.')[0])
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)

        tissues_list = [prob_csf_path, prob_wm_path, prob_gm_path]
        register_image(fixed_image_path, moving_image_path, fixed_mask_path, output_folder, parameter_files, tissues_list)

def register_image(fixed_image_path, moving_image_path, fixed_mask_path, output_folder, parameter_files, prob_path):
    fixed_image = itk.imread(fixed_image_path, itk.F)
    fixed_mask = itk.imread(fixed_mask_path, itk.UC)
    moving_image = itk.imread(moving_image_path,itk.F)

    parameter_object = itk.ParameterObject.New()
    for param_file in parameter_files:
        parameter_object.AddParameterFile(param_file)

    elastix_object = itk.ElastixRegistrationMethod.New(fixed_image, moving_image)
    elastix_object.SetFixedMask(fixed_mask)
    elastix_object.SetParameterObject(parameter_object)
    elastix_object.SetLogToConsole(False)
    elastix_object.SetOutputDirectory(output_folder)

    # Perform the registration
    elastix_object.UpdateLargestPossibleRegion()

    # Save the result of registration
    result_image = elastix_object.GetOutput()
    itk.imwrite(result_image, os.path.join(output_folder, "image_registered.nii.gz"))

    # Save the transformation parameter files
    result_transform_parameters = elastix_object.GetTransformParameterObject()

    transform_param_files = find_transform_parameter_files(output_folder)
    print(transform_param_files)
    for tissue in prob_path:
        apply_transformation(tissue, output_folder, transform_param_files)


def apply_transformation(moving_label_path, output_folder, transform_param_files):
    # Read the moving label image and doing everything with SITK from this point 'cause I don't fucking understand how to load sucessive transfromparameters with ITK
    moving_label = sitk.ReadImage(moving_label_path)

    transformix_transform = sitk.TransformixImageFilter()
    transformix_transform.SetMovingImage(moving_label)
    transformix_transform.SetOutputDirectory(output_folder)
    #in https://github.com/SuperElastix/SimpleElastix/issues/341 says you need to Set the first parameter file and add the rest
    transformix_transform.SetTransformParameterMap(sitk.ReadParameterFile(transform_param_files[0]))

    # Load and add the subsequent parameter maps
    for param_file in transform_param_files[1:]:
        transformix_transform.AddTransformParameterMap(sitk.ReadParameterFile(param_file))

    transformix_transform.Execute()

    result_label_transformix = transformix_transform.GetResultImage()
    label_name = os.path.basename(moving_label_path).replace('.nii.gz', '_transformed.nii.gz')
    sitk.WriteImage(result_label_transformix, os.path.join(output_folder, label_name))

def find_transform_parameter_files(output_folder):
    #because we cannot iterate on result_parameter object so this is necessary to pass the different
    #transformparameter
    parameter_files = []
    i = 0
    while True:
        param_file = os.path.join(output_folder, f"TransformParameters.{i}.txt")
        if os.path.exists(param_file):
            parameter_files.append(param_file)
            i += 1
        else:
            break
    return parameter_files

def apply_threshold(image_data, threshold=0.45):
    return (image_data > threshold).astype(np.int)

def process_images(testing_folder, segmentation_strategy='th'):#used for label propagation
    images_folder = os.path.join(testing_folder, 'testing-images')
    labels_folder = os.path.join(testing_folder, 'testing-labels')
    our_folder = os.path.join(testing_folder, 'testing-our')
    mni_folder = os.path.join(testing_folder, 'testing-mni')
    dice_scores = {}
    tissue_maps = {}

    for image_file in os.listdir(images_folder):
        base_name = image_file.split('.')[0]
        additional_string = "_label_prop"

        label_path = os.path.join(labels_folder, f"{base_name}_3C.nii.gz")
        label_img = nib.load(label_path)
        label_data = label_img.get_fdata()

        for folder, prefix in [(our_folder, 'probabilistic'), (mni_folder, 'mni')]:
            save_filename = f"{prefix}{base_name}{additional_string}.png"
            # Load probability maps
            csf_path = os.path.join(folder, base_name, f"{prefix}_CSF_transformed.nii.gz")
            gm_path = os.path.join(folder, base_name, f"{prefix}_GM_transformed.nii.gz")
            wm_path = os.path.join(folder, base_name, f"{prefix}_WM_transformed.nii.gz")

            csf_data = nib.load(csf_path).get_fdata()
            gm_data = nib.load(gm_path).get_fdata()
            wm_data = nib.load(wm_path).get_fdata()


            if segmentation_strategy == 'th':
                tissue_map = np.argmax(np.array([apply_threshold(csf_data), apply_threshold(gm_data), apply_threshold(wm_data)]), axis=0) + 1
            elif segmentation_strategy == 'mp':
                probabilities = np.array([csf_data, wm_data, gm_data])
                max_prob = np.max(probabilities, axis=0)

                # Set voxels to zero where all probabilities are below the threshold
                tissue_map = np.where(max_prob > 0.3, np.argmax(probabilities, axis=0)+1, 0)
            else:
                raise ValueError("Invalid segmentation strategy") 
            
            # Ensure same shape for Dice score calculation
            if tissue_map.shape != label_data.shape:
                min_shape = np.minimum(tissue_map.shape, label_data.shape)
                padded_tissue_map = np.zeros(label_data.shape)
                padded_tissue_map[:min_shape[0], :min_shape[1], :min_shape[2]] = tissue_map[:min_shape[0], :min_shape[1], :min_shape[2]]
                tissue_map = padded_tissue_map
            if prefix == 'probabilistic':
                tissue_maps[base_name] = np.where(max_prob > 0.3, probabilities,0)
            # Calculate Dice score for each tissue type
            dice_csf = dice_score_per_tissue(label_data, tissue_map, 1)
            dice_wm = dice_score_per_tissue(label_data, tissue_map, 2)
            dice_gm = dice_score_per_tissue(label_data, tissue_map, 3)
            if prefix == 'probabilistic':
                dice_scores[base_name] = {'Position_CSF': dice_csf, 'Position_WM': dice_wm, 'Position_GM': dice_gm}

            print(f"Dice score for CSF in {base_name} ({prefix}): {dice_csf}")
            print(f"Dice score for WM in {base_name} ({prefix}): {dice_wm}")
            print(f"Dice score for GM in {base_name} ({prefix}): {dice_gm}")
            plot_comparison(tissue_map, label_data, title1="Segmented Tissue", title2="Ground Truth", slice_nums=(136, 142, 163), save_filename=save_filename)
    return tissue_maps, dice_scores

def dice_score_per_tissue(true_labels, pred_labels, tissue_value):
    true_labels_tissue = (true_labels == tissue_value).astype(int)
    pred_labels_tissue = (pred_labels == tissue_value).astype(int)
    intersection = np.sum(true_labels_tissue * pred_labels_tissue)
    return 2. * intersection / (np.sum(true_labels_tissue) + np.sum(pred_labels_tissue)) if (np.sum(true_labels_tissue) + np.sum(pred_labels_tissue)) > 0 else 0

def dice_score_per_tissue_jaccard(true_labels, pred_labels, tissue_value):
    true_labels_tissue = (true_labels == tissue_value).astype(int)
    pred_labels_tissue = (pred_labels == tissue_value).astype(int)
    intersection = np.sum(true_labels_tissue * pred_labels_tissue)
    dice_score = 2. * intersection / (np.sum(true_labels_tissue) + np.sum(pred_labels_tissue)) if (np.sum(true_labels_tissue) + np.sum(pred_labels_tissue)) > 0 else 0
    jaccard_index = intersection / (np.sum(true_labels_tissue) + np.sum(pred_labels_tissue) - intersection) if (np.sum(true_labels_tissue) + np.sum(pred_labels_tissue) - intersection) > 0 else 0
    return dice_score, jaccard_index

def normalize_image(image_data, old_min, old_max, new_min=113, new_max=2000):
    nonzero_mask = image_data > 0
    normalized_image = np.zeros_like(image_data, dtype=float)
    normalized_image[nonzero_mask] = (((image_data[nonzero_mask] - old_min) / (old_max - old_min)) 
                                      * (new_max - new_min) + new_min)
    return normalized_image

def plot_comparison(data1, data2, title1="Tissue Map", title2="Ground Truth", slice_nums=(10, 50, 100), save_filename=None):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Comparison of Tissue Map and Ground Truth")

    # Define a color map for the tissue types
    cmap = mcolors.ListedColormap(['black', 'blue', 'green', 'red'])
    bounds = [0, 1, 2, 3, 4]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    legend_patches = [mpatches.Patch(color=cmap(i), label=label) for i, label in enumerate(['Background', 'CSF', 'WM', 'GM'])]

    for i, orientation in enumerate(["Axial", "Sagittal", "Coronal"]):
        for j, (data, title) in enumerate(zip([data1, data2], [title1, title2])):
            slice_data = np.rot90(data[slice_nums[i], :, :] if orientation == "Axial" else
                                  data[:, slice_nums[i], :] if orientation == "Sagittal" else
                                  data[:, :, slice_nums[i]], k=-1)
            axes[j, i].imshow(slice_data, cmap=cmap, norm=norm, origin="lower")
            axes[j, i].set_title(f"{title} - {orientation} Slice")
            axes[j, i].axis("off")

    plt.figlegend(handles=legend_patches, loc='lower center', ncol=4)

    if save_filename:
        plt.savefig(save_filename, bbox_inches='tight')
        plt.close(fig) 
    else:
        plt.show()

def segment_tissue_with_gmm(masked_image_data, mask):
    data = masked_image_data.flatten().reshape(-1, 1)
    gmm = GaussianMixture(n_components=3, covariance_type='full',
                          means_init=[[470.82], [1367.77 ], [885.63 ]],
                          precisions_init=[[[1/((229.09)**2)]], [[1/((187.59)**2)]], [[1/((183.41)**2)]]])
    gmm.fit(data)
    tissue_probs = gmm.predict_proba(data)
    tissue_map = np.argmax(tissue_probs, axis=1).reshape(masked_image_data.shape) + 1
    tissue_map = apply_mask(tissue_map, mask)
    return tissue_map, tissue_probs

def segment_tissue_with_intensity(masked, csf_model, wm_model, gm_model, affine, output_filename, mask):
    # Calculate unnormalized probabilities for each tissue type
    csf_prob_unnorm = norm.pdf(masked, csf_model['mean'], csf_model['sd'])
    wm_prob_unnorm = norm.pdf(masked, wm_model['mean'], wm_model['sd'])
    gm_prob_unnorm = norm.pdf(masked, gm_model['mean'], gm_model['sd'])

    #to debug print("CSF Probabilities Sample:", gm_prob_unnorm[113:200, 113:200])
    unnormalized_probs = np.stack([csf_prob_unnorm, wm_prob_unnorm, gm_prob_unnorm], axis=0)

    # Normalize the probabilities so they sum to 1 across the tissue types
    prob_sums = np.sum(unnormalized_probs, axis=0)
    normalized_probs = unnormalized_probs / prob_sums
    tissue_map = np.argmax(normalized_probs, axis=0) + 1 

    # Create a NIfTI image from the tissue map to debug
    tissue_map_img = nib.Nifti1Image(tissue_map.astype(np.int16), affine)

    # Save the tissue map as a NIfTI file
    nib.save(tissue_map_img, output_filename)
    tissue_map = apply_mask(tissue_map, mask)

    return tissue_map, normalized_probs

def segment_with_csv(masked_image_data, csv_file):
    df = pd.read_csv(csv_file, index_col=0)

    min_intensity = int(np.round(df.index.min()))
    max_intensity = int(np.round(df.index.max()))

    # Interpolate to fill missing intensity values
    df = df.reindex(range(min_intensity, max_intensity + 1)).interpolate()

    # Segment each pixel based on the CSV probabilities
    tissue_map = np.zeros_like(masked_image_data, dtype=int)

    for intensity in np.unique(masked_image_data):
        if intensity in df.index:
            tissue_type = np.argmax(df.loc[intensity].values) + 1 
            tissue_map[masked_image_data == intensity] = tissue_type

    return tissue_map

def apply_mask(image_data, mask_data):
    return image_data * mask_data

def process_images_with_mask(testing_folder): #used for tissuesegmentation
    csf_model = {'mean': 470.82, 'sd': 229.09}
    wm_model = {'mean': 1367.77, 'sd': 187.59}
    gm_model = {'mean': 885.63, 'sd': 183.41}

    images_folder = os.path.join(testing_folder, 'testing-images')
    labels_folder = os.path.join(testing_folder, 'testing-labels')
    masks_folder = os.path.join(testing_folder, 'testing-mask')
    dice_scores = {}
    tissue_maps = {}

    for image_file in os.listdir(images_folder):
        base_name = image_file.split('.')[0]
        additional_string = "_intensity"
        save_filename = f"{base_name}{additional_string}.png"
        image_path = os.path.join(images_folder, image_file)
        label_path = os.path.join(labels_folder, f"{base_name}_3C.nii.gz")
        mask_path = os.path.join(masks_folder, f"{base_name}_1C.nii.gz")

        image_data = nib.load(image_path).get_fdata()
        label_data = nib.load(label_path).get_fdata()
        mask_data = nib.load(mask_path).get_fdata()

        masked_image_data = apply_mask(image_data, mask_data)
        old_min, old_max = np.min(masked_image_data), np.max(masked_image_data)
        normalized_image_data = normalize_image(masked_image_data, old_min, old_max)

        #tissue_map, _ = segment_tissue_with_gmm(normalized_image_data, mask_data)
        tissue_map, _ = segment_tissue_with_intensity(normalized_image_data, csf_model, wm_model, gm_model, nib.load(image_path).affine, "test.nii.gz", mask_data)
        #tissue_map = segment_with_csv(normalized_image_data, 'tissue_type_histograms.csv')
        tissue_maps[base_name] = _

        # adding jacard to our scores just to verify
        dice_csf, jaccard_csf = dice_score_per_tissue_jaccard(label_data, tissue_map, 1)
        dice_wm, jaccard_wm = dice_score_per_tissue_jaccard(label_data, tissue_map, 2)
        dice_gm, jaccard_gm = dice_score_per_tissue_jaccard(label_data, tissue_map, 3)
        dice_scores[base_name] = {'Intensity_CSF': dice_csf, 'Intensity_WM': dice_wm, 'Intensity_GM': dice_gm}

        print(f"Dice/Jaccard score for {base_name}: CSF - {dice_csf}/{jaccard_csf}, WM - {dice_wm}/{jaccard_wm}, GM - {dice_gm}/{jaccard_gm}")

        plot_comparison(tissue_map, label_data, title1="Segmented Tissue", title2="Ground Truth", slice_nums=(136, 142, 163), save_filename=save_filename)
    return tissue_maps, dice_scores


def export_dice_scores(testing_folder, pos, intens, comb):
    position_scores = pos
    intensity_scores = intens
    combined_scores = comb

    data = []
    for image_number in position_scores:
        row = {
            'Image': image_number,
            'Position_CSF': position_scores[image_number]['Position_CSF'],
            'Position_WM': position_scores[image_number]['Position_WM'],
            'Position_GM': position_scores[image_number]['Position_GM'],
            'Intensity_CSF': intensity_scores.get(image_number, {}).get('Intensity_CSF', 0),
            'Intensity_WM': intensity_scores.get(image_number, {}).get('Intensity_WM', 0),
            'Intensity_GM': intensity_scores.get(image_number, {}).get('Intensity_GM', 0),
            'Combined_CSF': combined_scores.get(image_number, {}).get('Combined_CSF', 0),
            'Combined_WM': combined_scores.get(image_number, {}).get('Combined_WM', 0),
            'Combined_GM': combined_scores.get(image_number, {}).get('Combined_GM', 0)
        }
        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv('dice_scores_summary.csv', index=False)
    print(df)

def process_combined_images(testing_folder):
    tissue_maps_pos, _p = process_images(testing_folder, segmentation_strategy='mp')
    tissue_maps_int, _i = process_images_with_mask(testing_folder)
    
    combined_dice_scores = {}
    masks_folder = os.path.join(testing_folder, 'testing-mask')
    
    for base_name, tissue_map_pos in tissue_maps_pos.items():
        tissue_map_int = tissue_maps_int[base_name]
        additional_string = "_combined"
        save_filename = f"{base_name}{additional_string}.png"

        # Multiplication of tissue maps from both methods
        combined_tissue_map = tissue_map_pos*tissue_map_int
        tissue_map_final = np.argmax(combined_tissue_map, axis=0) + 1

        label_path = os.path.join(testing_folder, 'testing-labels', f"{base_name}_3C.nii.gz")
        mask_path = os.path.join(masks_folder, f"{base_name}_1C.nii.gz")
        mask_data = nib.load(mask_path).get_fdata()

        tissue_map_final = apply_mask(tissue_map_final, mask_data)
        label_data = nib.load(label_path).get_fdata()

        dice_csf = dice_score_per_tissue(label_data, tissue_map_final, 1)
        dice_wm = dice_score_per_tissue(label_data, tissue_map_final, 2)
        dice_gm = dice_score_per_tissue(label_data, tissue_map_final, 3)

        combined_dice_scores[base_name] = {'Combined_CSF': dice_csf, 'Combined_WM': dice_wm, 'Combined_GM': dice_gm}
        plot_comparison(tissue_map_final, label_data, title1="Segmented Tissue", title2="Ground Truth", slice_nums=(136, 142, 163), save_filename=save_filename)
    export_dice_scores('testing', _p, _i, combined_dice_scores)
    return combined_dice_scores