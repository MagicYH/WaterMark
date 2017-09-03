#

import os
import random
import tensorflow as tf
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

        writer = tf.python_io.TFRecordWriter(self._recordPath)
        for imgName in os.listdir(self._sourcePath):
            path = self._sourcePath + "/" + imgName
            img = Image.open(path)
            # if img.mode != 'RGBA':
            #     img = img.convert('RGBA')
            img = img.convert('L')
            [width, height] = img.size
            if width < self._tWidth or height < self._tHeight:
                continue
            wNum = int(round((width - self._tWidth) / 100)) + 1
            hNum = int(round((height - self._tHeight) / 100)) + 1
            for x in range(wNum):
                for y in range(hNum):
                    regin = (x * 100, y * 100, x * 100 + self._tWidth, y * 100 + self._tHeight)
                    tmpImg = img.crop(regin)
                    count = count + 1

                    label = [1, 0]
                    if count % 2 == 1:
                        tmpImg = self._addWaterRandPos(tmpImg)
                        label = [0, 1]
                    imgRaw = tmpImg.tobytes()
                    
                    if count % 200 == 1:
                        tmpImg.save(self._outPath + "/" + str(count) + ".png")
                    
                    example = tf.train.Example(features=tf.train.Features(feature={
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
                        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[imgRaw]))
                    }))
                    writer.write(example.SerializeToString())  #serialize example into string
        print("Create %d images" % count)
        writer.close()

    def _addWaterRandPos(self, sImg):
        # Random size, 15% ~ 30%
        # percent = 15.0 + random.randint(0, 15)
        percent = 100
        
        x1 = random.randint(0, self._tWidth - self._markWidth - 1)
        y1 = random.randint(0, self._tHeight - self._markHeight - 1)

        x2 = x1 + self._markWidth
        y2 = y1 + self._markHeight

        return ImageHelper.AddWaterWithImg(sImg, self._markImg, x1, y1, x2, y2)