# -*- coding: utf-8 -*-

import tensorflow as tf
from src.model.Identify2 import GetModel
from PIL import Image
import numpy

tWidth = 81
tHeight = 24
dw = tWidth / 3
dh = tHeight / 3
img = Image.open("data/type1/19.png")
width, height = img.size
wNum = int(round((width - tWidth) / dw)) + 1
hNum = int(round((height - tHeight) / dh)) + 1

data = []
batch = []
count = 0
for x in range(wNum):
    for y in range(hNum):
        regin = (x * dw, y * dh, x * dw + tWidth, y * dh + tHeight)
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


input = numpy.reshape(batch, (100, tWidth, tHeight, 1))


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


s = sess.run([y_conv], feed_dict={x: input})

print(s)
print(s[0][40])
print(s[0][58])
print(s[0][82])
print(s[0][75])

