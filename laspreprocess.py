#!/usr/bin/env python3 

import rasterio as rs
import numpy as np
import argparse
from laspy import file
from sys import argv
from scipy.spatial import KDTree
from time import time

def utm2img(coords, tf):
 
    #import pdb; pdb.set_trace()

#    tfMatInv = np.linalg.pinv(tfMat)
#    imgCoords = tfMatInv @ coords.T

    imgCoords = rs.transform.rowcol(tf, coords[:,0], coords[:,1])

    return imgCoords

def pointCloud2ImageCoords(fname, tf, Intensity=False, sf=100):
    #import pdb; pdb.set_trace()
    
    f = file.File(fname, mode='r')
    
    if Intensity:
        
        pointCloud = np.vstack([f.X, f.Y, f.Z, f.Intensity]).T / sf
        row, col = utm2img(pointCloud[:, :2], tf)
        pointCloud = np.column_stack((row, col))
    else:
        pointCloud = np.vstack([f.X, f.Y, f.Z]).T / sf
        row, col = utm2img(pointCloud[:, :2], tf)
        pointCloud = np.column_stack((row, col))
    f.close()
	
    return pointCloud

def label(pointCloud, img, fname):
    
    #band1 = img.read(1)
    start = time()
    nRows, nCols = img.shape
    
    x = np.arange(nCols)   
    y = np.arange(nRows)

    centers = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
    
    tree = KDTree(centers)
    
    dist, ind = tree.query(pointCloud)
    stop = time()
    print(str(stop-start))
    colandRow = centers[ind].astype(int)
    
    labelInd = (colandRow[:, 1], colandRow[:, 0])
    
    Label = img[labelInd]
    np.savetxt(fname, Label)
    
    print('Labels have been saved to {}'.format(fname))
    
    return Label

def relabel(pcloudfname, labelFname):
	pcloud = file.File(pcloudfname, mode='rw')
	label = np.loadtxt(labelFname)

	pcloud.Classification = label.astype(np.int8)
	
	pcloud.close()

	print('Point cloud has been relabeled.')

	return



#with rs.open('/home/wei/project/pointnet.pytorch/data/2018_Release_Phase1/GT/2018_IEEE_GRSS_DFC_GT_TR.tif') as img:
#    gray = img.read(1)
#    #import pdb; pdb.set_trace()
#    pointCloud = pointCloud2ImageCoords('/home/wei/project/data/2018_Release_Phase1/LiDAR PointCloud/C1/272056_3289689.las', img.transform)
#    Labels = label(pointCloud, gray[:,:1192], 'C1P1')
#    print(np.unique(Labels, return_counts = True))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='lasfile preprocess')
	parser.add_argument('--relabel', action='store_const', const=relabel, help='relabel las file')
	parser.add_argument('--pcloud', type=str, help='point cloud file location')
	parser.add_argument('--label', type=str, help='label file location')
	
	
	args = parser.parse_args()

	if args.relabel:
		relabel(args.pcloud, args.label)


