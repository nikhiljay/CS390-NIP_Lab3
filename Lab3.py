
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import random
from PIL import Image
from scipy.optimize import fmin_l_bfgs_b
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import warnings
import os

random.seed(1618)
np.random.seed(1618)
tf.random.set_seed(1618)
tf.compat.v1.disable_eager_execution()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

CONTENT_IMG_PATH = "memorial_mall.jpg"           #TODO: Add this.
STYLE_IMG_PATH = "starry_night.jpg"             #TODO: Add this.

CONTENT_IMG_H = 500
CONTENT_IMG_W = 500

STYLE_IMG_H = 500
STYLE_IMG_W = 500

CONTENT_WEIGHT = 0.1    # Alpha weight.
STYLE_WEIGHT = 1.0      # Beta weight.
TOTAL_WEIGHT = 1.0

TRANSFER_ROUNDS = 1

#=============================<Helper Fuctions>=================================
'''
TODO: implement this.
This function should take the tensor and re-convert it to an image.
'''
def deprocessImage(img):
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype('uint8')
    return img

def gramMatrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

#========================<Loss Function Builder Functions>======================

def styleLoss(style, gen):
    numFilters = 3
    ss = K.sum(K.square(gramMatrix(style) - gramMatrix(gen)))
    return ss / (4. * (numFilters^2) * ((STYLE_IMG_H * STYLE_IMG_W)^2))

def contentLoss(content, gen):
    return K.sum(K.square(gen - content))

def totalLoss(content_loss, style_loss):
    return CONTENT_WEIGHT * content_loss + STYLE_WEIGHT * style_loss

#=========================<Pipeline Functions>==================================

def getRawData():
    print("   Loading images.")
    print("      Content image URL:  \"%s\"." % CONTENT_IMG_PATH)
    print("      Style image URL:    \"%s\"." % STYLE_IMG_PATH)
    cImg = load_img(CONTENT_IMG_PATH)
    tImg = cImg.copy()
    sImg = load_img(STYLE_IMG_PATH)
    print("      Images have been loaded.")
    return ((cImg, CONTENT_IMG_H, CONTENT_IMG_W), (sImg, STYLE_IMG_H, STYLE_IMG_W), (tImg, CONTENT_IMG_H, CONTENT_IMG_W))

def preprocessData(raw):
    img, ih, iw = raw
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = img.resize(size=(ih, iw))
    img = img_to_array(img)
    img = img.astype("float64")
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

'''
TODO: Allot of stuff needs to be implemented in this function.
First, make sure the model is set up properly.
Then construct the loss function (from content and style loss).
Gradient functions will also need to be created, or you can use K.Gradients().
Finally, do the style transfer with gradient descent.
Save the newly generated and deprocessed images.
'''
def styleTransfer(cData, sData, tData):
    print("   Building transfer model.")
    contentTensor = K.variable(cData)
    styleTensor = K.variable(sData)
    genTensor = K.placeholder((1, CONTENT_IMG_H, CONTENT_IMG_W, 3))
    inputTensor = K.concatenate([contentTensor, styleTensor, genTensor], axis=0)
    model = vgg19.VGG19(include_top=False, weights="imagenet", input_tensor=inputTensor)
    outputDict = dict([(layer.name, layer.output) for layer in model.layers])
    print("   VGG19 model loaded.")
    loss = 0.0
    styleLayerNames = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
    contentLayerName = "block5_conv2"

    print("   Calculating content loss.")
    contentLayer = outputDict[contentLayerName]
    contentOutput = contentLayer[0, :, :, :]
    genOutput = contentLayer[2, :, :, :]
    content_loss = contentLoss(contentOutput, genOutput)
    loss += content_loss   #TODO: implement.

    print("   Calculating style loss.")
    style_loss = 0.0
    for layerName in styleLayerNames:
        styleLayer = outputDict[layerName]
        styleOutput = styleLayer[1, :, :, :]
        genOutput = styleLayer[2, :, :, :]
        style_loss += styleLoss(styleOutput, genOutput)   #TODO: implement.
    loss += style_loss / len(styleLayerNames)
    loss += totalLoss(content_loss, style_loss)   #TODO: implement.

    # TODO: Setup gradients or use K.gradients().
    grads = K.gradients(loss, genTensor)[0]
    kFunction = K.function([genTensor], [loss, grads])

    class Evaluator(object):
        def __init__(self):
            self.loss_value = None
            self.grads_values = None

        def loss(self, x):
            assert self.loss_value is None
            x = x.reshape((1, CONTENT_IMG_H, CONTENT_IMG_W, 3))
            outs = kFunction([x])
            loss_value = outs[0]
            grad_values = outs[1].flatten().astype('float64')
            self.loss_value = loss_value
            self.grad_values = grad_values
            return self.loss_value
                
        def grads(self, x):
            assert self.loss_value is not None
            grad_values = np.copy(self.grad_values)
            self.loss_value = None
            self.grad_values = None
            return grad_values

    evaluator = Evaluator()

    x = tData.flatten()
    print("   Beginning transfer.")
    for i in range(TRANSFER_ROUNDS):
        print("   Step %d." % i)
        #TODO: perform gradient descent using fmin_l_bfgs_b.
        x, tLoss, info = fmin_l_bfgs_b(evaluator.loss, x, fprime=evaluator.grads, maxiter = 1300, maxfun = 20)
        print("      Loss: %f." % tLoss)
        img = x.copy().reshape((CONTENT_IMG_H, CONTENT_IMG_W, 3))
        img = deprocessImage(img)
        saveFile = "style_output.jpg"
        #Uncomment when everything is working right.
        Image.fromarray(img).save(saveFile)
        print("      Image saved to \"%s\"." % saveFile)
    print("   Transfer complete.")

#=========================<Main>================================================

def main():
    print("Starting style transfer program.")
    raw = getRawData()
    cData = preprocessData(raw[0])   # Content image.
    sData = preprocessData(raw[1])   # Style image.
    tData = preprocessData(raw[2])   # Transfer image.
    styleTransfer(cData, sData, tData)
    print("Done. Goodbye.")

if __name__ == "__main__":
    main()
