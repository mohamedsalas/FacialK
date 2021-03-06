from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tensorflow.compat.v1 as tf
from scipy import misc
from skimage.transform import rescale, resize
import cv2
import numpy as np
import facenet
import detect_face
import os
import time
import pickle
global str
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--model-cfg', type=str, default='./yolo_cfg/yolov3-face.cfg',
                    help='path to config file')
parser.add_argument('--model-weights', type=str,
                    default='./yolo_weights/yolov3-wider_16000.weights',
                    help='path to weights of model')
args = parser.parse_args()

print('Creating networks and loading parameters')
# 
net = cv2.dnn.readNetFromDarknet(args.model_cfg, args.model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():

        minsize = 20  
        threshold = [0.6, 0.7, 0.7]  
        factor = 0.709 
        margin = 44
        frame_interval = 3
        batch_size = 1000
        image_size = 182
        input_image_size = 160

        print('Loading feature extraction model')
        modeldir = './models/20180408-102900/20180408-102900.pb'
        facenet.load_model(modeldir)
        #with tf.gfile.FastGFile("./models/20180402-114759/20180402-114759.pb", 'rb') as f:
        #  graph_def = tf.GraphDef()
        #  graph_def.ParseFromString(f.read())
        #  _ = tf.import_graph_def(graph_def, name='')

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        classifier_filename = './myclassifier/youssef.pkl'
        classifier_filename_exp = os.path.expanduser(classifier_filename)
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile)
            print('load classifier file-> %s' % classifier_filename_exp)

        video_capture = cv2.VideoCapture(0)
        c = 0

        print('Start Recognition!')
        prevTime = 0
        while True:
            ret, frame = video_capture.read()

            

            curTime = time.time()    
            timeF = frame_interval

            if (c % timeF == 0):
                find_results = []

                if frame.ndim == 2:
                    frame = facenet.to_rgb(frame)
                frame = frame[:, :, 0:3]
               

                # 
                blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT), [0, 0, 0], 1, crop=False)

                #
                net.setInput(blob)

                # Runs the forward pass to get output of the output layers
                outs = net.forward(get_outputs_names(net))

                #
                bounding_boxes = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
                nrof_faces = len(bounding_boxes)
                

                if nrof_faces > 0:
                   
                    img_size = np.asarray(frame.shape)[0:2]

                   
                    bb = np.zeros((nrof_faces,4), dtype=np.int32)

                    for i in range(nrof_faces):
                        emb_array = np.zeros((1, embedding_size))

                        bb[i][0] = bounding_boxes[i][0]
                        bb[i][1] = bounding_boxes[i][1]
                        bb[i][2] = bounding_boxes[i][2]
                        bb[i][3] = bounding_boxes[i][3]

                        if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                            print('face is inner of range!')
                            continue

                       
                        cropped = (frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                        print("{0} {1} {2} {3}".format(bb[i][0], bb[i][1], bb[i][2], bb[i][3]))
                        cropped = facenet.flip(cropped, False)
                        scaled = (resize(cropped, (image_size, image_size)))
                        scaled = cv2.resize(scaled, (input_image_size,input_image_size),
                                            interpolation=cv2.INTER_CUBIC)
                        scaled = facenet.prewhiten(scaled)
                        scaled_reshape = (scaled.reshape(-1,input_image_size,input_image_size,3))
                        feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}

                        emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)

                        predictions = model.predict_proba(emb_array)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                        cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                        text_x = bb[i][0]
                        text_y = bb[i][3] + 20
                        t_x=bb[i][2]
                        t_y=bb[i][1]+20

                        
                        result_names = class_names[best_class_indices[0]] if best_class_probabilities[0] > 0.9 else "Unknown"
                        
                        print(best_class_probabilities[0])
                        cv2.putText(frame,result_names, (text_x, text_y) , cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, (0, 0, 255), thickness=1, lineType=2)
                else:
                    print('Unable to align')

            sec = curTime - prevTime
            prevTime = curTime
            fps = 1 / (sec)
            str = 'FPS: %2.3f' % fps
            text_fps_x = len(frame[0]) - 150
            text_fps_y = 20
            cv2.putText(frame, str, (text_fps_x, text_fps_y),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), thickness=1, lineType=2)
            
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        
        cv2.destroyAllWindows()
