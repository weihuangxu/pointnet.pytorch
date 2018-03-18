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
    #import pdb; pdb.set_trace()
    #band1 = img.read(1)
    start = time()
    nRows, nCols = img.shape
    mincol = min(pointCloud[:,1])
    minrow = min(pointCloud[:,0])
    x = np.arange(nCols)
    x = x + mincol
    y = np.arange(nRows)
    y = y +  minrow
    

    #centers = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
    centers = np.transpose([np.tile(y, len(x)), np.repeat(x, len(y))])

    
    tree = KDTree(centers)
    
    dist, ind = tree.query(pointCloud)
    stop = time()
    print(str(stop-start))
    rowandCol = centers[ind].astype(int)
    
    
    labelInd = (rowandCol[:, 0] - minrow, rowandCol[:, 1] - mincol)
    
    Label = img[labelInd]
    np.savetxt(fname, Label)
    
    print('Labels have been saved to {}'.format(fname))
    
    return Label

def relabel(pcloudfname, labelFname):
	pcloud = file.File(pcloudfname, mode='rw')
	label = np.loadtxt(labelFname)

	pcloud.Classification = label.astype(np.int8)
	import pdb; set_trace()
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
	parser.add_argument('--labelpath', type=str, help='label file location')
	parser.add_argument('--label', action='store_const', const=label, help='label las file, save as textfile')
	parser.add_argument('--laspath', type=str, help='las file location')
	parser.add_argument('--labelout', type=str, help='label output name')
	
	
	args = parser.parse_args()

	if args.relabel:
		relabel(args.pcloud, args.labelpath)

	elif args.label:
		with rs.open('/home/wei/project/pointnet.pytorch/data/2018_Release_Phase1/GT/2018_IEEE_GRSS_DFC_GT_TR.tif') as img:
  	 		gray = img.read(1)
    #import pdb; pdb.set_trace()
   			pointCloud = pointCloud2ImageCoords(args.laspath, img.transform)
   			#import pdb; pdb.set_trace()
   			mincol = min(pointCloud[:,1])
   			maxcol = max(pointCloud[:,1])
   			minrow = min(pointCloud[:,0])
   			maxrow = max(pointCloud[:,0])
   			Labels = label(pointCloud, gray[minrow:maxrow,mincol:maxcol], args.labelout)
   			print(np.unique(Labels, return_counts = True))
   			print(np.unique(gray[minrow:maxrow+1 ,mincol:maxcol+1], return_counts=True))

