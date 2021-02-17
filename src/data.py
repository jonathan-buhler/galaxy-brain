from astroNN.datasets import galaxy10

# Downloads/loads dataset to/from ~/.astroNN/datasets
# Corporate says to ignore the obnoxious environment warnings
images, labels = galaxy10.load_data()

print(images)
print(labels)
