# from src.data.input.DataCreator2 import DataCreator
from src.data.input.DataCreator3 import DataCreator
from src.trainter2 import Train
creator = DataCreator("image/source", "image/mark/taptap_small.png", "data/type3")
creator.create()

# Train("data/data.tfrecord", 160, 120)

# Train("data/data.tfrecord")