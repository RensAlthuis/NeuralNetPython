import matplotlib.image as img
import csv

#This file just processes the image data and transforms it to matrices and labels

def readImg(list):
    X = []
    k = 0
    for id in list:
        X.append([])
        im = img.imread("./hasy-data/v2-" + '{:05}'.format(id) + ".png")
        for i in range(len(im)):
            for j in range(len(im[i])):
                X[k].append(im[i][j][0])
        k = k+1
    return X


def readOutput(name):
    outfile = open(name)
    reader = csv.reader(outfile, delimiter=',')
    y = []
    header = reader.__next__()
    for row in reader:
        if (int(row[1]) >= 70 and int(row[1]) <= 79) or (int(row[1]) >= 31 and int(row[1]) <= 57):
            y.append([row[0][13:-4], int(row[1]), row[2]] )
    return header, y

