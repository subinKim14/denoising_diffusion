'''
celebA.py : 
Apply some proper transformation (such as crop or reshape) on celebA images clean images to acquire test samples.
Then, it saves test samples in the directory.
Please modify the path to configuration file and the path to directory.
'''
import os
import json
import torchvision.transforms as T
import torchvision.datasets
import matplotlib.pyplot as plt
import numpy

'''
class for celebA 128 x 128 test data setup
'''
class obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, obj(b) if isinstance(b, dict) else b)

class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)


class CropTransform:
    def __init__(self, bbox):
        self.bbox = bbox

    def __call__(self, img):
        return img.crop(self.bbox)

def get_test_data(conf):

    if conf.dataset.name == 'celeba':
        transform = T.Compose(
            [
                CropTransform((25, 50, 25 + 128, 50 + 128)),
                T.Resize(128),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )
        transform_test = T.Compose(
            [
                CropTransform((25, 50, 25 + 128, 50 + 128)),
                T.Resize(128),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )
        '''
        train_set = torchvision.datasets.CelebA(conf.dataset.path,
                                                split='train',
                                                transform=transform,
                                                download=True)          
        '''

        test_set = torchvision.datasets.CelebA(conf.dataset.path,
                                                split='test',
                                                transform=transform_test,
                                                download=True)                            
 
    else:
        raise FileNotFoundError

    return test_set


if __name__ == '__main__':

    # modify the output directory to save celebA 128 x 128 images.
    output_dir = './data128x128'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # modify the configuration file path here.
    path_to_config = '../config/diffusion_celeba.json'
    with open(path_to_config, 'r') as f:
        conf = json.load(f)

    conf = obj(conf)

    test_set = get_test_data(conf)

    # modify the number of images to save.
    numimgs = 500

    for i in range(numimgs):
        img = test_set[i][0].cpu().numpy()
        img = (img+1)/2
        img = img.transpose(1, 2, 0)

        plt.imsave(os.path.join(output_dir, f'{i}.png'), img)

