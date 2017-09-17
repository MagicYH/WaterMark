#coding=utf-8

import os
import random
import tensorflow as tf
import numpy
import cv2
from scipy import signal
from PIL import Image
from src.data.img.ImageHelper import ImageHelper


class DataCreator():
    def __init__(self, sourcePath, markPath, outPath):
        self._cwd = os.getcwd() + "/"
        self._sourcePath = self._cwd + sourcePath
        self._markPath = self._cwd + markPath
        self._markImg = Image.open(self._markPath)
        if self._markImg.mode != 'RGBA':
            self._markImg = self._markImg.convert('RGBA')
        [self._markWidth, self._markHeight] = self._markImg.size
        self._outPath = self._cwd + outPath
        self._recordPath = self._outPath + "/data.tfrecord"
        self._tWidth = 81
        self._tHeight = 24
    
    def create(self):
        """ Create learning data

        Args: 
            sourcePath: Folder path of source images
            markPath: water mark file path
            width: width of created images
            height: height of created images
        
        Returns:
            None

        Raise:
            IOError          
        """
        count = 0
        cc = 0
        dWidth = int(self._tWidth)
        dHeight = int(self._tHeight)
        writer = tf.python_io.TFRecordWriter(self._recordPath)
        for imgName in os.listdir(self._sourcePath):
            path = self._sourcePath + "/" + imgName
            img = Image.open(path)
            # if img.mode != 'RGBA':
            #     img = img.convert('RGBA')
            img = img.convert('L')
            [width, height] = img.size
            height = int(405 * height / width);
            width = 405
            img = img.resize((width, height), Image.ANTIALIAS)
            wNum = int(round((width - self._tWidth) / dWidth)) + 1
            hNum = int(round((height - self._tHeight) / dHeight)) + 1
            for x in range(wNum):
                for y in range(hNum):
                    count = count + 1
                    x1 = x * dWidth
                    y1 = y * dHeight
                    if count % 2 == 1:
                        self._addWaterRandPos(img, x1, y1)

            kernel = [[-1, -1, -1],
                      [-1, 8, -1],
                      [-1, -1, -1]]
            imgData = numpy.array(img)
            imgData = signal.convolve2d(imgData, kernel, mode='same')
            imgData = (imgData - numpy.min(imgData))
            imgData = imgData * 255.0 / numpy.max(imgData)

            mean = imgData.mean() * 1.008
            imgData[imgData > mean] = 255
            imgData[imgData <=  mean] = 0

            # imgData = imgData.astype('uint8')
            # img = Image.fromarray(imgData, 'L')
            # tmpPath = self._outPath + "/tmp.png"
            # img.save(tmpPath)

            # cvImg = cv2.imread(tmpPath, 0)  # 读取灰度图像
            # hist = cv2.equalizeHist(cvImg)
            # cv2.imwrite(tmpPath, hist)

            # img = Image.open(tmpPath)
            # imgData = numpy.array(img)
            # mean = imgData.mean() * 1.3
            # imgData[imgData > mean] = 255
            # imgData[imgData <= mean] = 0

            imgData = imgData.astype('uint8')
            img = Image.fromarray(imgData, 'L')

            for x in range(wNum):
                for y in range(hNum):
                    cc = cc + 1
                    x1 = x * dWidth
                    y1 = y * dHeight
                    regin = (x * dWidth, y * dHeight, x * dWidth + self._tWidth, y * dHeight + self._tHeight)
                    tmpImg = img.crop(regin)

                    imgRaw = tmpImg.tobytes()
                    tmpImgData = numpy.array(tmpImg)
                    
                    labelFlag = 0
                    label = [1, 0]
                    mean = tmpImgData.mean();
                    std = numpy.std(tmpImgData);
                    if cc % 2 == 1 and mean < 108 and std < 125:
                        labelFlag = 1
                        label = [0, 1]
                    
                    if cc % 20 == 1:
                        print("index: %d, mean: %d, std: %d" % (cc, mean, std))
                        tmpImg.save(self._outPath + "/" + str(cc) + "_" + str(labelFlag) + ".png")
                    
                    example = tf.train.Example(features=tf.train.Features(feature={
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
                        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[imgRaw]))
                    }))
                    writer.write(example.SerializeToString())  #serialize example into string
        print("Create %d images" % count)
        writer.close()

    def _addWaterRandPos(self, sImg, x, y):
        # Random size, 15% ~ 30%
        # percent = 15.0 + random.randint(0, 15)
        percent = random.randint(95, 105)
        width = int(self._markWidth * percent / 100)
        height = int(self._markHeight * percent / 100)
        
        x1 = x + random.randint(0, self._tWidth - width - 1)
        y1 = y + random.randint(0, self._tHeight - height - 1)

        x2 = x1 + width
        y2 = y1 + height

        return ImageHelper.AddWaterWithImg(sImg, self._markImg, x1, y1, x2, y2)