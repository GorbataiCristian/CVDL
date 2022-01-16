import cv2
import numpy as np
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from PIL import Image

# from resizeimage  import resizeimage

# model = MobileNetV2(weights="imagenet")
# size = 224, 224
# # print(model.summary())
# data = np.empty((1, 224, 224, 3))
# im = Image.open('00003.jpg')
# # im.show()
#
# # im = resizeimage.resize_thumbnail(im, size)
# im = im.resize((224, 224))
# # im.thumbnail(size, Image.ANTIALIAS)
# # im.show()
#
# # img = imread()
# # img = tf.image
# data[0] = im
# data = preprocess_input(data)
# # print(data.shape)
# predict = model.predict(data)
#
# for name, desc, score in decode_predictions(predict, top=20)[0]:
#   print(f' {desc}, -- {score}')
from images import get_images

model = MobileNetV2(weights="imagenet")
size = (224, 224)
valid_classes = ["minivan", "limousine", "cab", "minibus", "pickup", "sports_car", "tow_truck", "convertible",
                 "moving_van", "police_van", "jeep", "ambulance", "beach_wagon"]


def mobile_net_v2_predict(img):
    img = img.resize(size)
    data = np.empty((1, 224, 224, 3))
    data[0] = img
    data = preprocess_input(data)
    predict = model.predict(data)

    prd = decode_predictions(predict, top=1)[0][0][1]
    if prd in valid_classes:
        return True, prd

    return False, prd


# images = get_images(1000)
# results = []
# positives = 0
# for image in images.vehicle_images:
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     im_pil = Image.fromarray(image)
#     res = mobile_net_v2_predict(im_pil)
#     results.append(res)
#     if res[0]:
#         positives += 1
#     # else:
#     #     cv2.imshow('image', image)
#     #     cv2.waitKey(0)
