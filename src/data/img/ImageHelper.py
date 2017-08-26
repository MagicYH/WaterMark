import os
from PIL import Image

# Image helper, a helper class to preprocess images

class ImageHelper():
    pass

    @staticmethod
    def AddWater(sPath, wPath, x1, y1, x2, y2):
        """Add water mark to a image

        Args:
            sPath: source image file path
            wPath: water image file path
            x1, y1, x2, y2: position water will be place into source image, when this value small than 1, it will mean percentage

        Returns:
            image that mark with water
        Raise:
            IOError
            ValueError
        """

        sImg = Image.open(sPath)
        wImg = Image.open(wPath)
        sSize = sImg.size
        waterWidth = x2 - x1
        waterHeight = y2 - y1

        # parameter check
        if waterWidth > sSize[0] or waterHeight > sSize[1]:
            raise ValueError("Invalid area config")
        if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0:
            raise ValueError("Invalid area config")
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid area config")

        if x1 < 1 and x2 < 1 and y1 < 1 and y2 < 1:
            x1 = round(sSize[0] * x1)
            x2 = round(sSize[0] * x2)
            y1 = round(sSize[1] * y1)
            y2 = round(sSize[1] * y2)
        
        tImg = wImg.resize((x2 - x1, y2 - y1), Image.ANTIALIAS)
        box = (x1, y1, x2, y2)
        rImg = sImg.paste(tImg, box)
        sImg.show()
        exit()
        print(tImg)
        print(sImg)
        print(x1, y1, x2, y2)
        print(rImg)
        exit()
        return rImg