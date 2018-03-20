import glob
import numpy as np
from skimage.transform import resize
from matplotlib.pyplot import imshow, imread
import random

class CelebA():
    def __init__(self, 
                 path_imgs = './CelebA/imgs/', 
                 path_attributes = './CelebA/list_attr_celeba.txt',
                 batchsize = 32,
                 flipping=True, 
                 imshape = None
                ):
        assert isinstance(path_imgs, str), 'please provide a path for the images folder'
        assert isinstance(path_attributes, str), 'please provide a path for the attributes file'
        assert isinstance(batchsize, int), 'error: invalid batch size provided'
        assert isinstance(flipping, bool), 'error: invalid flip value'
        
        self.path_imgs = path_imgs
        self.path_attributes = path_attributes
        
        self.num_attributes = 40
        with open(path_attributes) as file:
            self.attributes = file.read().strip().split()
            self.num_images = int(self.attributes[0])
            attributes_names = self.attributes[1:self.num_attributes+1]
            self.attributes = self.attributes[self.num_attributes+1:]
        self.images_dict = {}
        for i in range(self.num_images):
            attributes = {}
            line = self.attributes[i*(1+self.num_attributes):(i+1)*(1+self.num_attributes)]
            for i in range(len(line)-1):
                attributes[attributes_names[i]] = (1+int(line[i+1]))//2
            self.images_dict[line[0]] = attributes
        
        self.id_list = list(self.images_dict.keys())
        # sets the batchsize
        self.batchsize = batchsize
        # whether or not to flip the images horizontally at random
        self.flipping = flipping
        # flipping probability
        self.flip_prob = 0.5
        # if None, do not resize. Else, reshape to imsize
        self.imshape = imshape
        
#     the image path of an image, i.e. the image's dictionary key, 
#     is simply the path of the imag folder plus the id
    def image_file(self,image_id):
        image = imread(self.path_imgs+image_id)
        shape = image.shape
        d = (shape[0]-shape[1])//2
        assert d>0, 'error: wrong image format, should have aspect ratio>1'
        #crop the images to 178x178
        image = image[d:-d,:,:]
        if not self.imshape is None:
            assert isinstance(self.imshape[0], int), 'error: wrong image size specified'
            image = resize(image, self.imshape)
        #normalize the images to [-1,1]
        image = (image.astype(float)/255-0.5)*2
        if self.flipping:
            if random.random()<self.flip_prob:
                image = image[:,::-1,:]
        return image
        
    def batch(self, attributes_list):
        batchsize = self.batchsize
        img_ids = random.sample(self.id_list, batchsize)
        images = np.stack([self.image_file(idx) for idx in img_ids])
        attributes = np.stack([np.array([self.images_dict[idx][a] for a in attributes_list]) for idx in img_ids])
            
        return images, attributes
