# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 09:52:34 2018

@author: fuya
"""

import h5test
import sys

updateRate = 50
pending_file_path = "datasets/pending_face_test.h5"

def learn(inputImage, label):
    print(inputImage, pending_file_path)
    currentSize = h5test.append_image(pending_file_path, inputImage, label)
    if currentSize >= updateRate:
        updatedSize = h5test.update_dataset(pending_file_path)
        if updatedSize >=500:
            print("You need to update your training model")

''' 
h5test.create_pending_dataset(pending_file_path)       

n = 4
for i in range(3,3+n):
    filePath = 'images/happy_'+str(i)+'.jpg'
    print(filePath)
    learn(filePath, 1)
'''

image_path = sys.argv[1]
image_label = sys.argv[2]
learn(image_path, image_label)

def main():
    # print command line arguments
    image_path = sys.argv[1]
    image_label = sys.argv[2]
    learn(image_path, image_label)

if __name__ == "__main__":
    main()
    