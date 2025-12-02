import csv
import numpy as np
import matplotlib.pyplot as plt


def load_voltage(file,plot = False,size = 1024): # return array
    # load voltage as size*size*4 array
    array = []
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            for value in row[2:]:
                array.append((float(value)))
    array = np.array(array).reshape((size,size,4))
    # only retain values with circle with r = size // 2
    mask = np.zeros((size,size), dtype=bool)
    for i in range(size):
        for j in range(size):
            r2 = (i - size // 2) ** 2 + (j - size // 2) ** 2
            if r2 <= (size // 2) ** 2:
                mask[i, j] = True
    array[~mask] = np.nan
    if plot:
        plt.imshow(array[:,:,0])
    return array