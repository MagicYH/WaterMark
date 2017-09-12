# -*- coding: utf-8 -*-

import tensorflow as tf
from src.model.Identify2 import GetModel
from PIL import Image
from scipy import signal
import numpy
import random
from src.data.img.ImageHelper import ImageHelper

img = Image.open("image/source/00013_lighthouse_1280x800.jpg")
img = img.convert("L")
width, height = img.size
height = int(405 * height / width)
width = 405
img = img.resize((width, height), Image.ANTIALIAS)
mark = Image.open("image/mark/taptap_small.png")
mWidth, mHeight = mark.size
x1 = random.randint(0, width - mWidth - 1)
y1 = random.randint(0, height - mHeight - 1)
x2 = x1 + mWidth
y2 = y1 + mHeight
img = ImageHelper.AddWaterWithImg(img, mark, x1, y1, x2, y2)

kernel = [[0, -1, 0],
          [-1, 4, -1],
          [0, -1, 0]]
imgData = numpy.array(img)
imgData = signal.convolve2d(imgData, kernel, mode='same')
imgData = (imgData - numpy.min(imgData))
imgData = imgData * 255.0 / numpy.max(imgData)
imgData = imgData.astype('uint8')
img = Image.fromarray(imgData, 'L')
img.show()

tWidth = 81
tHeight = 24
dWidth = int(tWidth / 3)
dHeight = int(tHeight / 3)
wNum = int(round((width - tWidth) / dWidth)) + 1
hNum = int(round((height - tHeight) / dHeight)) + 1

data = []
batch = []
count = 0
for x in range(wNum):
    for y in range(hNum):
        regin = (x * dWidth, y * dHeight, x * dWidth + tWidth, y * dHeight + tHeight)
        tmpImg = img.crop(regin)
        tmpImg.save("data/res_view/" + str(count) + ".png")
        tmpData = numpy.array(tmpImg)
        tmpData = tmpData.reshape(tWidth, tHeight)
        data.append(tmpData)
        if len(data) == 100:
            batch.append(data)
            data = []
        count += 1

if len(data) != 0:
    while len(data) != 100:
        data.append(tmpData)
    batch.append(data)

input = []
for index in range(len(batch)):
    input.append(numpy.reshape(batch[index], (100, tWidth, tHeight, 1)))


modelSaveDir = "model/identify/"

sess = tf.Session()

# Create the model
x = tf.placeholder(tf.float32, [100, tWidth, tHeight, 1])

# Build the graph for the deep net
y_conv = GetModel(x)

## 这里是恢复graph
# saver = tf.train.import_meta_graph(modelSaveDir + 'train.model.meta')

saver = tf.train.Saver()
## 这里是恢复各个权重参数
saver.restore(sess, tf.train.latest_checkpoint(modelSaveDir))

count = 0
for index in range(len(input)):
    s = sess.run([y_conv], feed_dict={x: input[index]})
    for c in range(len(s[0])):
        count = count + 1
        if s[0][c][1] - s[0][c][0] > 3:
            print(s[0][c])
            print(count)
