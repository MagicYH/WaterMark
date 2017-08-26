import os
import tensorflow as tf
from PIL import Image


class DataCreator():
    
    @staticmethod
    def create(sourcePath, markPath, width, height):
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

        cwd = os.getcwd()
        cwd = cwd + "/"
        markPath = cwd + markPath
        waterImg = Image.open(markPath)
        writer = tf.python_io.TFRecordWriter("train.tfrecords")
        for imgName in os.listdir(sourcePath):
            path = cwd + sourcePath + "/" + imgName
            img = Image.open(path)
            