from src.model.Identify3 import Model

modelPath = "data/model/vgg"
model = Model(modelPath + "/model", modelPath + "/summary", modelPath + "/input")
# model.BuildData("image/mark/taptap_small.png", "image/source", modelPath + "/input")
model.Train(200)