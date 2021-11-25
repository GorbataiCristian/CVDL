import os

script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
from PIL import Images

RANDOM_IMAGES_RELATIVE_PATH = "/randoms/dataset_random"
VEHICLE_IMAGES_RELATIVE_PATH = "/cars"


class Images:
    def __init__(self, vehicle_images=None, other_images=None):
        if other_images is None:
            random_images = []
        if vehicle_images is None:
            vehicle_images = []
        self.vehicle_images = vehicle_images
        self.other_images = other_images

    def KFold(self, n_splits, shuffle=False):
        pass

    def load_all_random_images(self):
        self.other_images = []
        relative_path = script_dir + RANDOM_IMAGES_RELATIVE_PATH
        for image_name in os.listdir(relative_path):
            if image_name.endswith(".jpg") or image_name.endswith(".png"):
                self.other_images.append(load_random_image(relative_path, image_name))

    def load_all_vehicle_images(self):
        self.vehicle_images = []
        relative_path = script_dir + VEHICLE_IMAGES_RELATIVE_PATH
        for image_name in os.listdir(relative_path):
            if image_name.endswith(".jpg") or image_name.endswith(".png"):
                self.vehicle_images.append(load_random_image(relative_path, image_name))

    @staticmethod
    def load_image(relative_path, image_name):
        rel_path = relative_path + "/" + image_name
        img = Image.open(rel_path)
        return img


def main():
    images = Images()
    images.load_all_vehicle_images()
    images.load_all_random_images()

    print(len(images.vehicle_images))
    print(len(images.other_images))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
