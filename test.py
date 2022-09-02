import cv2 as cv
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import argparse
import sys
import os.path
import random
import os
import glob
import operator

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

image_size=128
num_channels=3
images = []
test_path='test_data'

# =============================================================================
# outputFile = sys.argv[2]
# 
# # Opening frames
# cap = cv.VideoCapture(sys.argv[1])
# 
# vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 15, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
# 
# width = int(round(cap.get(cv.CAP_PROP_FRAME_WIDTH)))
# height = int(round(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
# 
# newHeight = int(round(height/2))
# =============================================================================

# Restoring the model
sess = tf.Session()
saver = tf.train.import_meta_graph('roadsurface-model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))

# Acessing the graph
graph = tf.get_default_graph()

#
y_pred = graph.get_tensor_by_name("y_pred:0")

#
x = graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y_true:0")
print(y_true.shape)
y_test_images = np.zeros((1, len(os.listdir('training_data'))))
print(y_test_images.shape)
# =============================================================================
# while cv.waitKey(1) < 0:
# =============================================================================

# =============================================================================
#     hasFrame, images = cap.read()
# 
#     finalimg = images
# 
#     if not hasFrame:
#         print("Classification done!")
#         print("Results saved as: ", outputFile)
#         cv.waitKey(3000)
#         break
# =============================================================================
classes = ['HRA','SD','SMA']
    
true = []
pred = []

for fields in classes:   
    index = classes.index(fields)
    print('Now going to read {} files (Index: {})'.format(fields, index))
    path = os.path.join(test_path, fields, '*g')
      
    files = glob.glob(path)
    
    y_true_cls = []
    y_pred_cls = []

    for fl in files:
        y_true_cls.append(fields)
        true.append(fields)
        images = cv.imread(fl)
        
        # Region Of Interest (ROI)
        height, width = images.shape[:2]
        newHeight = int(round(height/2))
        newWidth = int(round(25 * width / 100.0))
        
        images = images[newHeight+50:height, newWidth:width-50]
        images = cv.resize(images, (image_size, image_size), 0, 0, cv.INTER_LINEAR)
        
        # plt.imshow(images)
        # plt.axis('off')
        # plt.show()
        images = np.array(images, dtype=np.uint8)
        images = images.astype('float32')
        images = np.multiply(images, 1.0/255.0)
        
        x_batch = images.reshape(1, image_size, image_size, num_channels)

        #
        feed_dict_testing = {x: x_batch, y_true: y_test_images}
        result = sess.run(y_pred, feed_dict=feed_dict_testing)
        
        outputs = [result[0,0], result[0,1], result[0,2]]
        
        value = max(outputs)
        index = np.argmax(outputs)
        
        if index == 0:
            label = 'HRA'
            prob = str("{0:.2f}".format(value))
            y_pred_cls.append(label)
            pred.append(label)
            #color = (0, 0, 0)
        elif index == 1:
            label = 'SD'
            prob = str("{0:.2f}".format(value))
            y_pred_cls.append(label)
            pred.append(label)
            #color = (153, 102, 102)
        elif index == 2:
            label = 'SMA'
            prob = str("{0:.2f}".format(value))
            y_pred_cls.append(label)
            pred.append(label)
            #color = (0, 153, 255)
    
        #print(fl,' --> ',label,' --> ',prob)
    
    print(fields,' --> ',accuracy_score(y_true_cls, y_pred_cls))
    
    
cm = confusion_matrix(true,pred,labels = ['HRA','SD','SMA'])
print(accuracy_score(true,pred))
cm_array_df = pd.DataFrame(cm, index=classes, columns=classes)
sns.heatmap(cm_array_df, annot=True, annot_kws={"size": 12}, fmt='g') 

plt.show()


# =============================================================================
#     cv.rectangle(finalimg, (0, 0), (145, 40), (255, 255, 255), cv.FILLED)
#     cv.putText(finalimg, 'Class: ', (5,15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
#     cv.putText(finalimg, label, (70,15), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
#     cv.putText(finalimg, prob, (5,35), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
# 
# 
#     vid_writer.write(finalimg.astype(np.uint8))
# =============================================================================

sess.close()