from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)
        
        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')--> it focuses on train.txt
            
        # initializing paths and labels of the images
        images_paths = []
        labels = []
        
        # self.root is the root directory path. listdir() is a method that allows to extract lists from the directory path.
        # the output are lists (not in order).
        self.categories = os.listdir(self.root)
        self.categories.remove("BACKGROUND_Google") #remove the backgroung class

        for path in open(f"./Caltech101/{self.split}.txt"): # path is the file name in train.txt
          path = path.replace("\n", "") # remove the string \n in the file name of train.txt
          category = path.split("/")[0]   # split the string of the file name into a list, whose separator is /..the category is the first element of the list
                                          # before the / (ex. "accordion")
          if(category != "BACKGROUND_Google"):
            image_path = self.root + "/" + path # we define the path of every image 
            images_paths.append(image_path) # list of images_paths
            labels.append(self.categories.index(category)) # list of the categories of the images (=labels)

        self.data = pd.DataFrame(zip(images_paths, labels), columns = ["image_path", "label"]) # pandas.DataFrame is a 2D size mutable tabular data..it includes
        # a zip array (images_paths,labels) which are what's in the tabular, columns are the column labels to use for resulting frame.
        # self.data is my new dataset of paths and labels.

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''
        image_path = self.data["image_path"].iloc[index]  # Provide a way to access image and label via index
                                                          # Image should be a PIL Image
                                                          # label can be int
        image, label = pil_loader(image_path), self.data["label"].iloc[index]


        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.data) # Provide a way to get the length (number of elements) of the dataset
        return length
