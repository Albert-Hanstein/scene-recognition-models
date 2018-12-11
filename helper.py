import os
from PIL import Image


def from_folder(datadir_in):
    nameList_out = []
    for x in os.listdir(datadir_in):
        if x.endswith(".jpg"):
            nameList_out.append(x)
    return nameList_out


def walk(datadir):
    '''Function to record all files in the directory

    Args:
        datadir: Input directory that contains all the fil

    Returns:

    '''
    listx = []
    listy = []
    for root, dirs, files in os.walk(datadir):
        for file in files:
            if file.endswith(".jpg"):
                with open(os.path.join(root, file), 'rb') as fd:
                    im = Image.open(fd)
                    listx.append(im.size[0])
                    listy.append(im.size[1])

                # return {filename_in: [im.size[0], im.size[1]]}
    return listx, listy


def get_img_size(datadir_in):
    listx = []
    listy = []
    files = from_folder(datadir_in)
    for file in files:
        with open(os.path.join(datadir_in, file), 'rb') as fd:
            im = Image.open(fd)
            listx.append(im.size[0])
            listy.append(im.size[1])

    #return {filename_in: [im.size[0], im.size[1]]}
    return listx, listy


def get_cm_string(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [9])  # 5 is value length
    empty_cell = ' ' * columnwidth
    # Print header
    cm_string = '    ' + empty_cell + ' '
    for label in labels:
        cm_string += '%{0}s'.format(columnwidth) % label + ' '
    cm_string += '\n'
    # Print rows
    for i, label1 in enumerate(labels):
        cm_string += '    %{0}s'.format(columnwidth) % label1 + ' '
        for j in range(len(labels)):
            cell = "%{0}.0f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            cm_string += cell + ' '
        cm_string += '\n'

    return cm_string


if __name__ == "__main__":

    directory = '/home/geoffrey893/PycharmProjects/scene-recognition-models/training'

    images_list = from_folder(directory)
    # img_sizes = []
    # my_dict = {}

    # for images in images_list:
    #     print(images)
    #     my_dict.update(get_img_size(directory, images))
    listx, listy = walk(directory)

    print(max(listx))
    print(min(listx))
    print(max(listy))
    print(min(listy))



