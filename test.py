# from src.data.input.DataCreator2 import DataCreator
from src.data.input.DataCreator import DataCreator
from src.trainter2 import Train
creator = DataCreator("image/source", "image/mark/taptap.png", "data/type1", 200, 160)
creator.create()

# Train("data/data.tfrecord", 160, 120)

# Train("data/data.tfrecord")