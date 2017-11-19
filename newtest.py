from src.model.Identify5 import Model

modelPath = "data/model/vgg"
model = Model(modelPath + "/model", modelPath + "/summary", modelPath + "/input")
# model.BuildData("image/mark/taptap_small.png", "image/source/main", modelPath + "/input")
model.Train(10000000)
