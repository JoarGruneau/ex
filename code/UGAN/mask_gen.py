import numpy as np
import Image
import ImageDraw
import os
import shutil
import glob

num_classes = 10
skip_targets = ['2', '4', '5', '7', '8']
# 001 -> car 1378                                  0
# 002 -> truck (and somtimes schoolbusses) 307     1
# 004 -> tractor 190                               2
# 005 -> camping car 397                           3
# 007 ->  motorbike 4 <---------tooo few
# 008 -> bus/truck?? 3 <---------tooo few          1
# 009  -> Vans 101                                 4
# 010 -> other 204                                 5
# 011 -> pickup 954                                6
# 023 -> boat 171                                  7
# 031 -> plane 48                                  8
script_dir = os.path.dirname(__file__)
annotation_dir = os.path.join(os.getcwd(), 'dataset/Annotations1024/')


def create_mask(id, fill=1):
    annotation = os.path.join(os.getcwd(), 'dataset/Annotations1024/' + id +'.txt')
    if not os.path.exists(annotation):
        print (id)
        return False
    targets = open(annotation, 'r').read().splitlines()
    masks = {}
    img = Image.new('L', (1024, 1024))
    draw = ImageDraw.Draw(img)
    draw.polygon(((0,0), (0,1024), (1024, 1024), (1024, 0)), fill=fill)
    masks['9'] = [img, draw]

    for target in targets:
        target = target.split()
        poly = [(float(target[i]), float(target[i + 4])) for i in range(6, 10, 1)]

        if masks.get(target[3]) is None:
            img = Image.new('L', (1024, 1024))
            draw = ImageDraw.Draw(img)
            masks[target[3]] = [img, draw]

        masks[target[3]][1].polygon(poly, fill=fill)
        masks['9'][1].polygon(poly, fill=0)

    return masks


def create_ground_truth():
    save_path = os.path.join(os.getcwd(), 'ground_truth')
    image_path = os.path.join(os.getcwd(), 'dataset/Ve1024/')
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    for image_name in sorted(glob.glob(image_path +'/*_co.png')):
        id = os.path.basename(image_name).split('_')[0]
        print(id)
        masks = create_mask(id)
        if not masks:
            os.remove(image_name)
            continue

        for i in range(num_classes):
            if masks.get(str(i)) is None:
                Image.new('RGB', (1024, 1024)).save(
                    os.path.join(save_path, id+'_'+str(i)+'.png'))
            else:
                mask = np.asarray(masks[str(i)][0], dtype="float32")
                img = np.asarray(Image.open(image_name, 'r'), dtype="float32")
                for j in range(3):
                    np.multiply(img[:, :, j], mask, img[:, :, j])
                img = Image.fromarray(img.astype('uint8'))
                img.save(os.path.join(save_path, id+'_'+str(i)+'.png'))

def save_masks():
    save_path = os.path.join(os.getcwd(), 'ground_truth_masks')
    image_path = os.path.join(os.getcwd(), 'dataset/Ve1024/')
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    for image_name in sorted(glob.glob(image_path +'/*_co.png')):
        id = os.path.basename(image_name).split('_')[0]
        print(id)
        masks = create_mask(id, fill=255)

        for i in range(num_classes):
            if masks.get(str(i)) is None:
                Image.new('L', (1024, 1024)).save(
                    os.path.join(save_path, id+'_'+str(i)+'.png'))
            else:
                masks[str(i)][0].save(os.path.join(save_path, id+'_'+str(i)+'.png'))

def create_binary_mask(id, fill=1):
    annotation = os.path.join(os.getcwd(), 'dataset/Annotations1024/' + id +'.txt')
    if not os.path.exists(annotation):
        print (id)
        return False
    targets = open(annotation, 'r').read().splitlines()
    img = Image.new('L', (1024, 1024))
    draw = ImageDraw.Draw(img)

    for target in targets:
        target = target.split()
        if target[3] not in skip_targets:
            poly = [(float(target[i]), float(target[i + 4])) for i in range(6, 10, 1)]
            draw.polygon(poly, fill=fill)

    return img


def save_binary_masks():
    save_path = os.path.join(os.getcwd(), 'ground_truth_masks')
    image_path = os.path.join(os.getcwd(), 'images')
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)
    print sorted(glob.glob(image_path + '/*.png'))
    for image_name in sorted(glob.glob(image_path + '/*.png')):
        id = os.path.basename(image_name).split('.')[0]
        print(id)
        mask = create_binary_mask(id, fill=255)
        if mask:
            mask.save(os.path.join(save_path, id + '_mask.png'))
        else:
            os.remove(image_name)




# class Mask(object):
#     def __init__(self, targets):
#         self.size = 1024
#         self.img_id = targets
#         self.masks = {}
#
#         for target in targets:
#             if masks.get[li]
#
#
#
#     def get(self, class_mask):
#         if self.masks.get(class_mask) != None:
#             return self.masks[class_mask]
#         else:
#             return np.zeros(self.size, self.size)


if __name__ == "__main__":
    save_binary_masks()
    #create_ground_truth()
    # save_masks()

