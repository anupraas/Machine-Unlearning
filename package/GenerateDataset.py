import pickle
import numpy as np
from torchvision import datasets, transforms


class CustomDataset:
    def __init__(self):
        print('Initialized!')

    def get_dataset(self, name, filename):
        X, y = None, None
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,)),
                                        ])
        if name == 'cifar10':
            X, y = self.get_data(datasets.CIFAR10(root='data', download=True, train=True, transform=transform))
        elif name == 'mnist':
            X, y = self.get_data(datasets.MNIST(root='data', download=True, train=True, transform=transform))
        else:
            raise NotImplementedError('This dataset has not been implemented!')

        return X,y


    def get_data(self, dataset):

        feature_size = dataset[0][0].shape[0] * dataset[0][0].shape[1] * dataset[0][0].shape[2]
        X = np.zeros((len(dataset), feature_size))
        y = np.zeros(len(dataset))

        count = 0
        for i in range(len(dataset)):
            img, label = dataset[i]
            img = img.numpy().ravel()
            X[count, :] = img
            y[count] = label
            count += 1
        return X, y

# CustomDataset().get_dataset('cifar10', 'cifar10')
# CustomDataset().get_dataset('mnist', 'mnist')