import sys,os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
count = 0
for l in img:
    print("{0:>4}".format(l),end='')
    count = count + 1
    if count % 28 ==0:
        print()

# print(img)
# label = t_train[0]
# print(label)
# print(img.shape)
# img = img.reshape(28,28)
# print(img)
# print(img.shape)
# img_show(img)
