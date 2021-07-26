# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 20:10:58 2019

@author: Meet
"""

import csv
import cv2
import glob
import argparse
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET

# python voc_to_csv_conversion.py --dataset_path "F:/LEARN/DeepLearningDatasets/Datasets/PASCAL/Combined/" --dataset Train --output_path "./ann_files/" --include_bg_class

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", required=True, type=str, help="Path of dataset. Should not contain any spaces in the path.")
parser.add_argument("--dataset", required=True, type=str, help="Which dataset to be converted: train or test, Default: train")
parser.add_argument("--output_path", required=True, type=str, help="Location of the folder where the output file will be saved.")
parser.add_argument("--include_bg_class", action='store_true', help="Include background class at 0 index.")
args = parser.parse_args()

class_mapping = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5, 'car': 6, 'cat': 7, 'chair': 8,
                 'cow': 9, 'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15, 'sheep': 16, 
                 'sofa': 17, 'train': 18, 'tvmonitor': 19}
id_mapping = list(class_mapping.keys())


if __name__ == "__main__":
    # Dataset Structure should look like below.
    # Dataset folder
    #       - Train
    #               - JPEGImages
    #               - Annotations
    #       - Test
    #               - JPEGImages
    #               - Annotations
    
    if " " in args.dataset_path:
        print("Dataset path should not contain any spaces.")
        exit(0)
        
    # Note: here voc datasets have been merged.: Train Set of 2007 and 2012 are "train set" and Test set of 2007 and 2012 are "Test set".

    if str.lower(args.dataset) != 'train' and str.lower(args.dataset) != 'test':
        print("Please provide dataset option from 'Train' or 'Test' only.")
        exit(0)


    files = glob.glob(args.dataset_path + str.lower(args.dataset) + "/Annotations/*.xml")
    base_path = args.dataset_path + str.lower(args.dataset) + "/JPEGImages/"
    
    csvfile = open(args.output_path + "/voc_" + str.lower(args.dataset) + '_ann.csv', 'w', newline='\n')
    csv_writer = csv.writer(csvfile, delimiter=',')
    
    for f in tqdm(files):
        tree = ET.parse(f)
        root = tree.getroot()
        filename = root.find("filename").text
        #print(base_path + filename)
        img = cv2.imread(base_path + filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        ######################################################################
        # NOTE : MAKE SURE THAT THE BASE_PATH IS NOT HAVING ANY SPACES
        ######################################################################

        bbox_data = [base_path + filename, h, w]
        for item in root.findall("object"):
            objName = item.find("name").text        
            if args.include_bg_class:
                objId = class_mapping[objName] + 1
            else:
                objId = class_mapping[objName]			
            xmin = float(item.find("bndbox").find("xmin").text)     # In the original range of image
            ymin = float(item.find("bndbox").find("ymin").text)     # In the original range of image
            xmax = float(item.find("bndbox").find("xmax").text)     # In the original range of image
            ymax = float(item.find("bndbox").find("ymax").text)     # In the original range of image
            bbox_data.extend([int(xmin), int(ymin), int(xmax), int(ymax), int(objId)])
        csv_writer.writerow(bbox_data)
    csvfile.close()
    