from astroNN.datasets import galaxy10
from PIL import Image


def get_data():
    # Downloads/loads dataset to/from ~/.astroNN/datasets
    # Corporate says to ignore the obnoxious environmental warnings
    images, labels = galaxy10.load_data()

    test_image = images[0]
    img = Image.fromarray(test_image, mode='RGB')
    img.save('./src/out/test_image.jpg')

    return images, labels
