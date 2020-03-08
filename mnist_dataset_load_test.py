import os
import random
from mnist import MNIST

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# rounded_binarized - convert trainig dataset to arrays of 0,1
#mndata = MNIST('mnist_dataset', mode="rounded_binarized")
mndata = MNIST('mnist_dataset')

print('Loading training dataset...')
images, labels = mndata.load_training()
#images, labels = mndata.load_testing()


"""
while True:
    index = random.randrange(0, len(images))  # choose an index ;-)
    print(mndata.display(images[index], threshold = 0))
    #print(mndata.display(images[index], threshold = 0)) # for binarized
    os.system('cls')
"""

print(labels[1])
print(images[1])
print(mndata.display(images[1]))
#print(len(images[0])) # 784
