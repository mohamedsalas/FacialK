from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tensorflow as tf
from scipy import misc
import cv2
import numpy as np
import facenet
import os
import time
import pickle
from yolo.yolo import *

import random
from time import sleep

def main(args_list):
    sleep(random.random())

    args = args_list[0]

    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dataset = facenet.get_dataset(args.input_dir)

    # Add a random key to the filename to allow alignment using multiple processes
    random_key = np.random.randint(0, high=99999)
    bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)

    myYolo = YOLO(args_list[1])
    
    with open(bounding_boxes_filename, "w") as text_file:
        nrof_images_total = 0
        nrof_successfully_aligned = 0
        if args.random_order:
            random.shuffle(dataset)
        for cls in dataset:
            output_class_dir = os.path.join(output_dir, cls.name)
            if not os.path.exists(output_class_dir):
                os.makedirs(output_class_dir)
                if args.random_order:
                    random.shuffle(cls.image_paths)
            for image_path in cls.image_paths:
                nrof_images_total += 1
                filename = os.path.splitext(os.path.split(image_path)[1])[0]
                output_filename = os.path.join(output_class_dir, filename+'.png')
                print(image_path)
                if not os.path.exists(output_filename):
                    try:
                        img = misc.imread(image_path)
                    except (IOError, ValueError, IndexError) as e:
                        errorMessage = '{}: {}'.format(image_path, e)
                        print(errorMessage)
                    else:
                        if img.ndim<2:
                            print('Unable to align "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))
                            continue
                        if img.ndim == 2:
                            img = facenet.to_rgb(img)
                        img = img[:,:,0:3]
    
                        image = Image.fromarray(img)
                        _, bounding_boxes = myYolo.detect_image(image)
                        nrof_faces = len(bounding_boxes)

                        if nrof_faces>0:
                            det = bounding_boxes
                            det_arr = []
                            img_size = np.asarray(img.shape)[0:2]
                            if nrof_faces>1:
                                if args.detect_multiple_faces:
                                    for i in range(nrof_faces):
                                        det_arr.append(np.squeeze(det[i]))
                                else:
                                    bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                                    img_center = img_size / 2
                                    offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                                    offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                                    index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                                    det_arr.append(det[index,:])
                            else:
                                det_arr.append(np.squeeze(det))

                            for i, det in enumerate(det_arr):
                                det = np.squeeze(det)
                                bb = np.zeros(4, dtype=np.int32)
                                bb[0] = np.maximum(det[0]-args.margin/2, 0)
                                bb[1] = np.maximum(det[1]-args.margin/2, 0)
                                bb[2] = np.minimum(det[2]+args.margin/2, img_size[1])
                                bb[3] = np.minimum(det[3]+args.margin/2, img_size[0])
                                cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                                scaled = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear')
                                nrof_successfully_aligned += 1
                                filename_base, file_extension = os.path.splitext(output_filename)
                                if args.detect_multiple_faces:
                                    output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                                else:
                                    output_filename_n = "{}{}".format(filename_base, file_extension)
                                misc.imsave(output_filename_n, scaled)
                                text_file.write('%s %d %d %d %d\n' % (output_filename_n, bb[0], bb[1], bb[2], bb[3]))
                        else:
                            print('Unable to align "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))

                            
    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)
            

def get_arguments():
    parser1 = argparse.ArgumentParser()
    parser1.add_argument('--input_dir', type=str, help='Directory with unaligned images.', default = './unaligned_faces')
    parser1.add_argument('--output_dir', type=str, help='Directory with aligned face thumbnails.', default = './aligned_faces')
    parser1.add_argument('--image_size', type=int, help='Image size (height, width) in pixels.', default=160)
    parser1.add_argument('--margin', type=int, help='Margin for the crop around the bounding box (height, width) in pixels.', default=32)
    parser1.add_argument('--random_order', help='Shuffles the order of images to enable alignment using multiple processes.', action='store_true')
    parser1.add_argument('--gpu_memory_fraction', type=float, help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.5)
    parser1.add_argument('--detect_multiple_faces', type=bool, help='Detect and align multiple faces per image.', default=False)

    parser2 = argparse.ArgumentParser()
    parser2.add_argument('--model', type=str, default='yolo_weights/YOLO_Face.h5', help='path to model weights file')
    parser2.add_argument('--anchors', type=str, default='yolo_cfg/yolo_anchors.txt', help='path to anchor definitions')
    parser2.add_argument('--classes', type=str, default='yolo_cfg/face_classes.txt', help='path to class definitions')
    parser2.add_argument('--score', type=float, default=0.5, help='the score threshold')
    parser2.add_argument('--iou', type=float, default=0.45, help='the iou threshold')
    parser2.add_argument('--img-size', type=list, action='store', default=(416, 416), help='input image size')
    
    args1 = parser1.parse_args()
    args2 = parser2.parse_args()
    args = [args1, args2]

    return args;

if __name__ == '__main__':
    main(get_arguments())
