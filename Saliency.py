from matplotlib import cm
import matplotlib.pyplot as plt
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing.image import load_img





def loss(output):
    return (output[1][1], output[0][1])

def modifier(m):
    m.layers[-1].activation = tf.keras.activations.linear
    return m






img1 = load_img('path', target_size=(128, 128))
img2 = load_img('path', target_size=(128, 128))
img3 = load_img('path', target_size=(128, 128))

images = np.asarray([np.array(img1), np.array(img2),np.array(img3)])



X = preprocess_input(images)

model = tensorflow.keras.applications.VGG16(input_shape=(128,128), weights='imagenet', include_top=True)




saliency = Saliency(model, modifier, clone=False)

saliencymap = saliency(loss, X)
saliencymap = normalize(saliencymap)

f, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (6, 3))
for i in range(len(saliencymap)-1):
    ax[i].imshow(saliencymap[i], cmap='jet')
plt.tight_layout()
plt.show()
