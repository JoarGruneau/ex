import numpy as np
import Image
import ImageDraw
import os
import shutil
import glob
import cv2
import matplotlib.pyplot as plt

num_classes = 10
skip_targets = ['2', '5', '7', '8']
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
annotation_dir = os.path.join(os.getcwd(), 'vedai/Annotations512/')


def create_mask(id, fill=1):
    annotation = os.path.join(os.getcwd(), 'vedai/Annotations512/' + id +'.txt')
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

def create_binary_mask(id, fill=255, cutoff_count=0):
    annotation = os.path.join(os.getcwd(), 'vedai/Annotations512/' + id +'.txt')
    valid_targets = 0
    if not os.path.exists(annotation):
        print(annotation)
        print (id)
        return False
    targets = open(annotation, 'r').read().splitlines()
    img = Image.new('L', (512, 512))
    draw = ImageDraw.Draw(img)

    for target in targets:
        target = target.split()
        if target[3] not in skip_targets:
            valid_targets += 1
            poly = [(float(target[i]), float(target[i + 4])) for i in range(6, 10, 1)]
            # border_size=5
            # diff = [(-border_size,-border_size), (border_size, -border_size),
            #         (border_size, border_size), (-border_size, border_size)]
            # poly2 =[(poly[i][0] + diff[i][0], poly[i][1] + diff[i][1]) for i in range(len(poly))]
            # draw.polygon(poly2, fill=0)
            line_points = poly + [poly[0]]
            draw.polygon(poly, fill=fill)
            draw.line(line_points, fill=0, width=3)
            # for point in line_points:
            #     draw.ellipse((point[0] - 1, point[1] - 1, point[0] + 1, point[1] + 1), fill=0)

    if valid_targets < cutoff_count:
        return False
    else:
        return img


def save_binary_masks(cutoff_count=0):
    save_path = os.path.join(os.getcwd(), 'ground_truth_masks')
    image_path = os.path.join(os.getcwd(), 'vedai/Ve512_2')
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)
    for image_name in sorted(glob.glob(image_path + '/*.png')):
        id = os.path.basename(image_name).split('.')[0]
        print(id)
        mask = create_binary_mask(id, fill=255, cutoff_count=cutoff_count)
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
    if min(x, y) < 0 or max(x, y) >= component_image.shape[0]:
        return False
    # print not_labels
    return component_image[x, y] not in not_labels

def compare_and_update_distance(component_img, distance_and_label, x, y, i, j, sqare_expansion, not_labels):
    found_componet = compare_components(component_img, i, j, not_labels)
    if found_componet:
        distance = np.sqrt(np.power(x - i, 2) + np.power(y - j, 2))
        for k in range(2):
            if distance_and_label[k][0] > distance and component_img[i, j] != distance_and_label[k -1][1]:
                distance_and_label[k] = [distance, component_img[i, j], sqare_expansion]
    return distance_and_label


def get_distances(component_img, x, y, not_labels, cut_off):
    square_expansion = 1
    distance_and_label = [[float('inf'), None, None], [float('inf'), None, None]]
    while True:
        for i in range(x - square_expansion, x + square_expansion + 1):

            if i == x-square_expansion or i == x+square_expansion:
                for j in range(y - square_expansion, y + square_expansion + 1):
                    distance_and_label = compare_and_update_distance(component_img, distance_and_label, x, y, i, j,
                                                square_expansion, not_labels)

            else:
                for j in [y - square_expansion, y + square_expansion]:
                    # print ([i,j])
                    distance_and_label = compare_and_update_distance(component_img, distance_and_label, x, y, i, j,
                                                square_expansion, not_labels)
            if (square_expansion > cut_off):
                return distance_and_label[0][0], distance_and_label[1][0]
            if (max(distance_and_label[0][0], distance_and_label[1][0])*2 < square_expansion):
                return distance_and_label[0][0], distance_and_label[1][0]
        square_expansion += 1


def weight_function(component_img, x, y, not_labels, cut_off):
    distance_1, distance_2 = get_distances(component_img, x, y, not_labels, cut_off)
    value = np.exp(-np.divide(np.power(distance_1 + distance_2, 2), 2*3**2))
    return value


def compute_weight_map(data_path, train_path, mask_suffix, suffix, save_path):
    image_paths = get_image_paths(data_path, suffix)
    save_path = os.path.join(os.getcwd(), save_path)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)
    for image_path in list(list(glob.glob(os.path.join(train_path, '*'+suffix)))):
        print(image_path)
        id = os.path.basename(image_path)
        label = os.path.join(data_path, id.replace(suffix, mask_suffix))
        # print('label', label)
        img = cv2.imread(label, 0)
        # print(img)
        # print(img.shape)
        img = np.asarray(img)
        weight_map = np.zeros(img.shape)
        _, component_image = cv2.connectedComponents(img)
        for x in range(img.shape[0]):
            print (x)
            for y in range(img.shape[1]):
                if img[x, y] == 0:
                    weight_map[x, y] = weight_function(component_image, x, y, [0], 5)
        weight_map = weight_map * 255
        weight_map.astype(dtype=np.uint8)
        # weight_map_img = Image.fromarray(weight_map, 'L')
        cv2.imwrite(os.path.join(save_path, id + '_weight_map' + suffix), weight_map)

def plot_weight_maps(data_path, save_path, label_suffix, weight_suffix):
    image_paths = get_image_paths(data_path, label_suffix)
    weight_paths = get_image_paths(data_path, weight_suffix)
    save_path = os.path.join(os.getcwd(), save_path)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)
    for i in range(len(image_paths)):
        image_path = image_paths[i]
        weight_path = weight_paths[i]
        print (weight_path)
        id = os.path.basename(image_path).split('.')[0]
        print(id)
        img = cv2.imread(image_path, 0)
        img = np.asarray(img)/255
        print(np.amax(img))
        weight = np.asarray(cv2.imread(weight_path, 0), dtype=np.float32)/255
        print(np.amax(weight))
        weight_map = 1.0 - img + 2*img+10*weight
        # weight_map = weight_map/np.amax(weight_map)
        # weight_map = weight_map * 255
        # weight_map.astype(dtype=np.uint8)
        print(np.amin(weight_map))
        print(np.amax(weight_map))
        plt.imshow(weight_map, vmin = 1, vmax=10)
        plt.xticks([])
        plt.yticks([])
        plt.jet()
        plt.colorbar()
        plt.show()
        # # weight_map_img = Image.fromarray(weight_map, 'L')
        # cv2.imwrite(os.path.join(save_path, id + '_weight_map' + suffix), weight_map)


if __name__ == "__main__":
    #save_binary_masks()
    compute_weight_map('Potsdam/bin_lables_resized', 'Potsdam/RGB', '_label_mask.tif', '.tif', 'Potsdam/weight_maps')
    #postdam_mapping()
    #down_sample(6)
    # create_ground_truth()
    # save_masks()
   # plot_weight_maps('Potsdam/bin_labels_resized', 'Potsdam/weight_maps', '_label_mask_mask.tif', '_weight_map.tif')
