import glob
import os
import shutil

mapping = {
    '1': '0', '2': '1', '4': '2',
    '5': '3', '8': '1', '9': '4',
    '10': '5', '11': '6', '23': '7',
    '31': '8'}
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


def do_mapping():
    os.chdir('/home/joar/Documents/ex/code/UGAN/vedai/Annotations512/')
    for file_name in list(list(glob.glob('*.txt'))):
        file = open(file_name, 'r')
        targets = file.readlines()
        saved_targets = ""
        for i in range(len(targets)):
            line = targets[i].split()
            key = line[3]
            if key != '7':
                line[3] = mapping[line[3]]
                saved_targets = saved_targets + ' '.join(line) + '\n'
        file = open(file_name, 'w')
        file.write(saved_targets)


def count():
    count_map = {}
    os.chdir('/home/joar/Documents/ex/code/UGAN/vedai/Annotations512/')
    for file in list(list(glob.glob('*.txt'))):
        reader = open(file)
        for line in reader.readlines():
            line = line.split()
            #print(line)
            key = line[3]
            # print(key)
            if count_map.get(key) == None:
                print(key)
                count_map[key] = 1
            else:
                count_map[key] += 1
    print(count_map.items())

def number_above_cutoff(image_path, annotations_path, cutoff, copy = False, output_path =''):
    count = 0
    if copy:
        shutil.rmtree(output_path, ignore_errors=True)
        os.makedirs(output_path)
    for image_path in list(list(glob.glob(os.path.join(image_path, '*.png')))):
        id = os.path.basename(image_path).split('.')[0]
        annotation_path = os.path.join(annotations_path, id + '.txt')
        reader = open(annotation_path)
        lines = reader.readlines()
        image_count = 0
        for line in lines:
            if line.split()[3] not in skip_targets:
                image_count += 1
        if image_count >= cutoff:
            count += 1
            if copy:
                shutil.copyfile(image_path, os.path.join(output_path, os.path.basename(image_path)))
    print(count)

def clean_data():
    os.chdir('/home/joar/Documents/ex/code/UGAN/vedai/Annotations512/')
    for file_name in list(list(glob.glob('*.txt'))):
        file = open(file_name, 'r')
        targets = file.readlines()
        if wrong_format(targets):
            print(file_name)
            os.remove(file_name)

def wrong_format(lines):
    for i in range(len(lines)):
        line = lines[i].split()
        if len(line) != 14:
            return True
    return False

if __name__ == "__main__":
    number_above_cutoff('vedai/Ve512/validation', 'vedai/Annotations512', 5, copy=True, output_path='vedai/Ve512/val_small')
    #do_mapping()
    #clean_data()
    #count()
    #do_mapping()
    #count()