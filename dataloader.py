import os

class DataLoader:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.images_folder = os.path.join(data_folder, f"{data_folder}-images")
        self.labels_folder = os.path.join(data_folder, f"{data_folder}-labels")
        self.masks_folder = os.path.join(data_folder, f"{data_folder}-mask")

    def find_files(self):
        image_files = sorted(os.listdir(self.images_folder))
        label_files = sorted(os.listdir(self.labels_folder))
        mask_files = sorted(os.listdir(self.masks_folder))
        return image_files, label_files, mask_files

    def retrieve_data(self, number):
        image_filename = os.path.join(self.images_folder, f"{number}.nii.gz")
        label_filename = os.path.join(self.labels_folder, f"{number}_3C.nii.gz")
        mask_filename = os.path.join(self.masks_folder, f"{number}_1C.nii.gz")
        return image_filename, label_filename, mask_filename