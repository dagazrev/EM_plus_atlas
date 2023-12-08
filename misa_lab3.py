import atlas_utils as au
import dataloader as dl
import em as em1
import em2
import em3
import em4
import em_helper as eh
import os

#structure should be:
#atlases
#testing
#   testing-images
#   testing-labels
#   testing-mask

#testing-our and testing-mni should be created by registration methods
#forgot to put all the result images inside their corresponding folder.
#slplit the channels of the MNI atlas to have it separate as we had our previous atlas
input_file_path = "MNITemplateAtlas\\atlas.nii.gz"
au.split_nii_channels(input_file_path)

#compress files to save storage. Previous NIFTI were 150mb and in nii.gz they are less than 10mb
input_file_path2 = "atlases\mean_intensities_atlas_all.nii"
au.convert_nii_to_niigz(input_file_path2)

#registration and transformatio
data_folder = 'testing'
atlases_folder = os.path.join('atlases')
data_loader = dl.DataLoader(data_folder)
parameter_files = ['Par0033similarity.txt', 'Par0033bspline.txt']  # Replace with actual paths
au.register_image(data_loader, parameter_files, atlases_folder)

#no EM segmentation
au.process_images(data_folder, 'mp')
au.process_images_with_mask(data_folder)

#combined scores
combined_scores = au.process_combined_images('testing')
for base_name, scores in combined_scores.items():
    print(f"{base_name}: {scores}")

#EM
#probably a best merge of the different EM could have make us repeat less code
em_params = {
    'csf_model': {'mean': 470.82, 'sd': 229.09},
    'wm_model': {'mean': 1367.77, 'sd': 187.59},
    'gm_model': {'mean': 885.63, 'sd': 183.41}
}
dice_scores = em1.process_images_with_em(data_folder, em_params)
filename = 'em_dice_scores.csv'
eh.export_dice_scores(data_folder, dice_scores, filename)

dice_scores = em2.process_images_with_em(data_folder, em_params)
filename = 'em2_dice_scores.csv'
eh.export_dice_scores(data_folder, dice_scores, filename)

#in this one we forgot to automate so we need to verify lines 182 to 184 in em3.py file to change between MNI and probabilistic (our atlas)
#and we need to change the filename to 'em5_dice_scores.csv' and comment the rest to just run this piece of code again
######################################################
#Done this way to avoid saving a lot of nifties at expense of having more RAM or running blocks separately
######################################################

dice_scores = em3.process_images_probabilistic(data_folder, em_params)
filename = 'em3_dice_scores.csv'
eh.export_dice_scores(data_folder, dice_scores, filename)

dice_scores = em4.process_images_probabilistic(data_folder, em_params)
filename = 'em4_dice_scores.csv'
eh.export_dice_scores(data_folder, dice_scores)

