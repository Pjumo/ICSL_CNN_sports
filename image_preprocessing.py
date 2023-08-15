from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split

# 시작 디렉토리
path = './results'
catrgories = ['golf', 'soccer', 'tennis', 'valleyball']
num_classes = len(catrgories)

# 이미지의 크기 지정
image_h = 400
image_w = 400
pixels = image_w * image_h * 3

X = []
Y = []
for idx, cat in enumerate(catrgories):
    label = [0 for i in range(num_classes)]
    label[idx] = 1
    image_dir = os.path.join(path, cat)
    for filepath, _, filenames in os.walk(image_dir):
        for filename in filenames:
            if filename.endswith('.png'):
                img = Image.open(os.path.join(filepath, filename))
                img = img.convert('RGB')
                img = img.resize((image_w, image_h))
                data = np.asanyarray(img)
                X.append(data)
                Y.append(label)
X = np.array(X)
Y = np.array(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
xy = (X_train, X_test, Y_train, Y_test)
np.save("./results/sport.npy", xy)
