import yaml

with open('/Users/ranweisberg/PycharmProjects/Image Processing in OR Project/Image-Processing-in-OR-Project/Part 1/Data/labeled_image_data/data.yaml', 'r') as f:
    data = yaml.safe_load(f)

print(data)
