import os
import cv2

SCRIPT_DIR = os.path.dirname(__file__)  # <-- absolute dir the script is in
RANDOM_IMAGES_RELATIVE_PATH = r"\randoms\dataset_random"
VEHICLE_IMAGES_RELATIVE_PATH = r"\cars"


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

    def load_random_images(self, limit):
        print('Started loading the random images')
        self.other_images = []
        relative_path = SCRIPT_DIR + RANDOM_IMAGES_RELATIVE_PATH
        start_at = 200
        index = 0
        for image_name in os.listdir(relative_path):
            if start_at > index:
                index += 1
                continue
            if limit == 0:
                return
            if image_name.endswith(".jpg") or image_name.endswith(".png"):
                self.other_images.append(Images.load_image(relative_path, image_name))
                limit -= 1
        print('Finished loading the random images')

    def load_vehicle_images(self, limit):
        print('Started loading the vehicle images')
        self.vehicle_images = []
        relative_path = SCRIPT_DIR + VEHICLE_IMAGES_RELATIVE_PATH
        start_at = 200
        index = 0
        for image_name in os.listdir(relative_path):
            if start_at > index:
                index += 1
                continue
            if limit == 0:
                return
            if image_name.endswith(".jpg") or image_name.endswith(".png"):
                img = Images.load_image(relative_path, image_name)
                self.vehicle_images.append(img)
                limit -= 1

    @staticmethod
    def load_image(relative_path, image_name):
        rel_path = relative_path + "\\" + image_name
        return cv2.imread(rel_path)


def get_images(images_limit: int):
    images = Images()
    images.load_vehicle_images(images_limit)
    images.load_random_images(images_limit)
    return images


if __name__ == '__main__':
    images = get_images(75)
