import tensorflow as tf
import numpy as np
from PIL import Image
import os

NUM_CLASS = 1230




BASE_DIR = "../../data/ocr/capture"
file_list = os.listdir(BASE_DIR)
im = Image.open(os.path.join(BASE_DIR, file_list[10]))
im_array = np.array(im)
print(im_array.shape)

X = tf.constant(im_array)
X = X[tf.newaxis, ...]
resized_im_array = tf.image.resize_images(X, [60, 180])
rslt = tf.cast(resized_im_array, tf.int32)
sess = tf.Session()
out = sess.run(rslt)
new_im = Image.fromarray(np.uint8(out[0]))
new_im.save("new.jpg")
