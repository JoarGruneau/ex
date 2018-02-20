import glob
import os

mapping = {
    '1': '0', '2': '1', '4': '2',
    '5': '3', '8': '1', '9': '4',
    '10': '5', '11': '6', '23': '7',
    '31': '8'}

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
    os.chdir('/home/joar/Documents/ex/code/UGAN/dataset/Annotations1024/')
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
    os.chdir('/home/joar/Documents/ex/code/UGAN/dataset/Annotations1024/')
    for file in list(list(glob.glob('*.txt'))):
        reader = open(file)
        for line in reader.readlines():
            line = line.split()
            key = line[3]
            if count_map.get(key) == None:
                count_map[key] = 1
            else:
                count_map[key] += 1
    print(count_map.items())

def clean_data():
    os.chdir('/home/joar/Documents/ex/code/UGAN/dataset/Annotations1024/')
    for file_name in list(list(glob.glob('*.txt'))):
        file = open(file_name, 'r')
        targets = file.readlines()
        if wrong_format(targets):
            print(file_name)
            os.remove(file_name)

def wrong_format(lines):
    for i in range(len(lines)):
        line = lines[i].split()
        if len(line) < 14:
            return True
    return False

if __name__ == "__main__":
    clean_data()
    # count()
    # do_mapping()
    # count()