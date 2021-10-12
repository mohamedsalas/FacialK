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
from statistics import mode
from utils1.datasets import get_labels
from utils1.inference import detect_faces
from utils1.inference import draw_text
from utils1.inference import draw_bounding_box
from utils1.inference import apply_offsets
from utils1.inference import load_detection_model
from utils1.preprocessor import preprocess_input
from keras.models import load_model

parser = argparse.ArgumentParser()
parser.add_argument('--model-cfg', type=str, default='./yolo_cfg/yolov3-face.cfg',
                    help='path to config file')
parser.add_argument('--model-weights', type=str,
                    default='./yolo_weights/yolov3-wider_16000.weights',
                    help='path to weights of model')
args = parser.parse_args()
emotion_model_path = './models/emotion_model.hdf5'
emotion_labels = get_labels('fer2013')
emotion_window = []

print('Creating networks and loading parameters')
# Load YOLO V2 model.
net = cv2.dnn.readNetFromDarknet(args.model_cfg, args.model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        # pnet, rnet, onet = detect_face.create_mtcnn(sess, './models/')

        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        frame_interval = 3
        batch_size = 1000
        image_size = 182
        input_image_size = 160

        print('Loading feature extraction model')
        modeldir = './models/20170512-110547'
        facenet.load_model(modeldir)
        
        emotion_classifier = load_model(emotion_model_path)
        emotion_target_size = emotion_classifier.input_shape[1:3] 
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        classifier_filename = './myclassifier/my_classifier.pkl'
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

            # frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)

            curTime = time.time()    # calc fps
            timeF = frame_interval

            if (c % timeF == 0):
                find_results = []

                if frame.ndim == 2:
                    frame = facenet.to_rgb(frame)
                frame = frame[:, :, 0:3]
                #print(frame.shape[0])
                #print(frame.shape[1])

                # Use YOLO to get bounding boxes
                blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT), [0, 0, 0], 1, crop=False)

                # Sets the input to the network
                net.setInput(blob)

                # Runs the forward pass to get output of the output layers
                outs = net.forward(get_outputs_names(net))

                # Remove the bounding boxes with low confidence
                bounding_boxes = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
                nrof_faces = len(bounding_boxes)
                

                if nrof_faces > 0:
                    # det = bounding_boxes[:, 0:4]
                    img_size = np.asarray(frame.shape)[0:2]

                    # cropped = []
                    # scaled = []
                    # scaled_reshape = []
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

                        # cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                        # cropped[0] = facenet.flip(cropped[0], False)
                        # scaled.append(misc.imresize(cropped[0], (image_size, image_size), interp='bilinear'))
                        # scaled[0] = cv2.resize(scaled[0], (input_image_size,input_image_size),
                        #                        interpolation=cv2.INTER_CUBIC)
                        # scaled[0] = facenet.prewhiten(scaled[0])
                        # scaled_reshape.append(scaled[0].reshape(-1,input_image_size,input_image_size,3))
                        # feed_dict = {images_placeholder: scaled_reshape[0], phase_train_placeholder: False}
                        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        cropped = (frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                        cropped1 = gray_image[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2]]

                        
                        print("{0} {1} {2} {3}".format(bb[i][0], bb[i][1], bb[i][2], bb[i][3]))
                        cropped = facenet.flip(cropped, False)
                        scaled = (resize(cropped, (image_size, image_size)))
                        scaled = cv2.resize(scaled, (input_image_size,input_image_size),
                                            interpolation=cv2.INTER_CUBIC)
                        scaled1=cv2.resize(cropped1, (emotion_target_size))
                        scaled = facenet.prewhiten(scaled)
                        scaled_reshape = (scaled.reshape(-1,input_image_size,input_image_size,3))
                        feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}

                        emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)

                        scaled1 = preprocess_input(scaled1, True)
                        scaled1 = np.expand_dims(scaled1, 0)
                        scaled1 = np.expand_dims(scaled1, -1)
                        emotion_prediction = emotion_classifier.predict(scaled1)
                        emotion_probability = np.max(emotion_prediction)
                        emotion_label_arg = np.argmax(emotion_prediction)
                        emotion_text = emotion_labels[emotion_label_arg]
                        emotion_window.append(emotion_text)
                        #if len(emotion_window) > frame_window:
                        #       emotion_window.pop(0)
                        try:
                               emotion_mode = mode(emotion_window)
                        except:
                             continue

                        predictions = model.predict_proba(emb_array)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                        cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                        text_x = bb[i][0]
                        text_y = bb[i][3] + 20
                        t_x=bb[i][2]
                        t_y=bb[i][1]+20

                        # for H_i in HumanNames:
                        #     if HumanNames[best_class_indices[0]] == H_i:
                        result_names = class_names[best_class_indices[0]] if best_class_probabilities[0] > 0.85 else "Unknown"
                        #print(result_names)
                        print(best_class_probabilities[0])
                        cv2.putText(frame,result_names+'  '+emotion_mode, (text_x, text_y) , cv2.FONT_HERSHEY_COMPLEX_SMALL,
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
            # c+=1
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        # #video writer
        # out.release()
        cv2.destroyAllWindows()
