import numpy as np
from keras.applications.mobilenet_v2  import MobileNetV2, preprocess_input, decode_predictions
from imageio import imread
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

model = MobileNetV2(weights="imagenet")
size = (224, 224)
valid_classes = ["minivan", "limousine", "cab", "minibus", "pickup", "sports_car", "tow_truck", "convertible", "moving_van", "police_van", "jeep", "ambulance"]
def mobile_net_v2_predict(img):
    img = img.resize(size)
    data = np.empty((1, 224, 224, 3))
    data[0] = img
    data = preprocess_input(data)
    predict = model.predict(data)

    print(decode_predictions(predict, top=1)[0])
    if decode_predictions(predict, top=1)[0] in valid_classes:
        return True

    return False
