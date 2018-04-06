import numpy as np
import Image
import ImageDraw
import os
import shutil
import glob
import cv2

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
    for image_name in sorted(glob.glob(image_path + '/*.png')):
        id = os.path.basename(image_name).split('.')[0]
        print(id)
        mask = create_binary_mask(id, fill=0)
        if mask:
            mask.save(os.path.join(save_path, id + '_mask.png'))
        else:
            os.remove(image_name)

def postdam_mapping():
    data_root=os.path.join(os.getcwd(), 'Potsdam')
    data_path = os.path.join(data_root, 'Labels')
    save_path=os.path.join(data_root, 'bin_lables')
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)
    size = 6000
    for image_name in sorted(glob.glob(data_path + '/*.tif')):
        id = os.path.basename(image_name).split('.')[0]
        print(id)
        img =Image.open(image_name, 'r').load()
        mask = Image.new('L', (size, size))
        mask_pixels=mask.load()
        for x in range(size):
            for y in range(size):
                if img[x, y] == (255, 255, 0):
                    mask_pixels[x, y]=255

        mask.save(os.path.join(save_path, id + '_mask.tif'))

def down_sample(factor):
    data_root=os.path.join(os.getcwd(), 'Potsdam')
    data_path = os.path.join(data_root, 'bin_lables')
    save_path=os.path.join(data_root, 'bin_lables_resized')
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)
    size = 6000
    for image_name in sorted(glob.glob(data_path + '/*.tif')):
        id = os.path.basename(image_name).split('.')[0]
        print(id)
        img =Image.open(image_name, 'r')
        img = np.asarray(img)
        print(img.shape)
        mask = Image.new('L', (size//factor, size//factor))
        mask_pixels=mask.load()
        for x in range(0, size, factor):
            for y in range(0, size, factor):
                # print("min")
                # print(np.amin(img))
                if np.median(img[x:x+factor, y:y+factor]) >= 255:
                    mask_pixels[y//factor, x//factor, ] = 255
        mask.save(os.path.join(save_path, id + '_mask.tif'))

def get_image_paths(relative_path, suffix):
    path = os.path.join(os.getcwd(), relative_path)
    return sorted(glob.glob(path + '/*' + suffix))

def compare_components(component_image, x, y, not_labels):
    if min(x, y) < 0 or max(x,y) >= component_image.shape[0]:
        return False
    # print not_labels
    return component_image[x,y] not in not_labels


def get_distances(component_img, x, y, not_labels, cut_off):
    square_expansion = 1
    distance_and_label = []
    found_componet=False
    while not found_componet:
        for i in range(x-square_expansion,x+square_expansion+1):
            if i == 0 or i == x + square_expansion:
                for j in range(y-square_expansion,y+square_expansion+1):
                    found_componet = compare_components(component_img, i, j, not_labels)
            else:
                for j in [y - square_expansion, y + square_expansion]:
                    found_componet = compare_components(component_img, i, j, not_labels)
            if found_componet:
                distance = np.sqrt(np.power(x - i, 2) + np.power(y - j, 2))
                if not distance_and_label or distance < distance_and_label[0]:
                    distance_and_label = [distance, component_img[i,j], square_expansion]
            if distance_and_label and distance_and_label[2]*2 > square_expansion:
                return True, distance_and_label[0], distance_and_label[1]
            if square_expansion > cut_off:
                return False, float('inf'), None

            # print coordinates_and_label
            # print square_expansion
            # print('x' if coordinates_and_label else 'y')
            # print(coordinates_and_label[3]*2 if coordinates_and_label else 'what')
            # print(coordinates_and_label[3]*2 <= square_expansion if coordinates_and_label else 'ada')
            # if coordinates_and_label and coordinates_and_label[3]*2 <= square_expansion:
            #     found_componet = True
        square_expansion += 1



def weight_function(component_img, x, y, not_labels, cut_off):
    found, distance_1, label_1 = get_distances(component_img, x, y, not_labels, cut_off)
    if found:
        __, distance_2, _ = get_distances(component_img, x, y, not_labels + [label_1], cut_off)
    else:
        distance_2 = distance_1
    value = np.exp(-np.divide(np.power(distance_1+distance_2, 2), 50))
    return value

def compute_weight_map(data_path, save_path, suffix):
    image_paths = get_image_paths(data_path, suffix)
    save_path=os.path.join(os.getcwd(), save_path)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)
    for image_path in image_paths:
        id = os.path.basename(image_path).split('.')[0]
        print(id)
        img =cv2.imread(image_path, 0)
        img = np.asarray(img)
        weight_map = np.zeros(img.shape)
        _, component_image = cv2.connectedComponents(img)
        for x in range(img.shape[0]):
            print(x)
            for y in range(img.shape[1]):
                if img[x, y] == 0:
                    weight_map[x,y]=weight_function(component_image, x, y, [0], 10)
        weight_map=weight_map*255
        weight_map.astype(dtype=np.uint8)
        # weight_map_img = Image.fromarray(weight_map, 'L')
        cv2.imwrite(os.path.join(save_path, id + '_weight_map' + suffix), weight_map)
        # weight_map_img.save(os.path.join(save_path, id + '_weight_map' + suffix))






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
    compute_weight_map('Potsdam/bin_labels_resized', 'Potsdam/weight_maps', '.tif')
    #postdam_mapping()
    # down_sample(6)
    #create_ground_truth()
    # save_masks()

