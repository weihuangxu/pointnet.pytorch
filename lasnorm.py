import argparse
import os.path
import sys
import glob
import numpy as np
from laspy.file import File

def normalization(objPath, outPath, labelPath, srcFile, intensity=True):
    objFiles = glob.glob(objPath + '*.las')

    for fname in objFiles:
        try:
            f = File(fname, mode='r')
            x_out = (f.X - np.mean(f.X)) / np.std(f.X)
            y_out = (f.Y - np.mean(f.Y)) / np.std(f.Y)
            z_out = (f.Z- np.mean(f.Z)) / np.std(f.Z)
            outFile = np.vstack((x_out, y_out, z_out))
            if intensity:
                i_out = (f.Intensity - np.mean(f.Intensity)) / np.std(f.Intensity)
                outFile = np.vstack((outFile, i_out))
            bname = os.path.basename(fname)
            name, ext = bname.split('.')
            np.savetxt(outPath+name+'.pts', outFile.T)
            np.savetxt(labelPath+name+'.seg', f.Classification.T)
            print('{}{}.pts and {}{}.seg have been saved.'.format(outPath, name, labelPath, name))
            f.close()

        except:
            continue
    
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Normalization')
    parser.add_argument('--fr', type=str, help='point clouds to be normalized')
    parser.add_argument('--to', type=str, help='where to save normalized point clouds')
    parser.add_argument('--label', type=str, help='where to save labels')
    parser.add_argument('--src', type=str, help='From which file generate these target point clouds')
    parser.add_argument('-i', action='store_true', default=False)
    args = parser.parse_args()
    normalization(args.fr, args.to, args.label, args.src, args.i)
