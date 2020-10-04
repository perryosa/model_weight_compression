import os
# from data_loader_end2end import loadfile
import tensorflow as tf
import numpy as np
import pdb
#import matplotlib.pyplot as plt
import math
from testing import testing
import pickle
import librosa
from scipy.io import wavfile
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
#from validation_denoiseLSTM import Validation
#from deepLSTMnetwork import network
max_length = 300
sr = 16000
nfft = 512
hop_length = 257
#Pink Noise Training
# data_path = '/home/perrryosa/Speech-Denoising-With-RNN/dataset_generate/pink_dataset/dataset_wind_narrow'
# dfc=os.listdir('/home/perrryosa/Speech-Denoising-With-RNN/dataset_generate/pink_dataset/dataset_wind_narrow/clean-audio')
# dfn=os.listdir('/home/perrryosa/Speech-Denoising-With-RNN/dataset_generate/pink_dataset/dataset_wind_narrow/noisy-audio')
# dfc.sort()
# dfn.sort()
# dfnoise=os.listdir('/home/perrryosa/Speech-Denoising-With-RNN/dataset_generate/wind_dataset_IITGN/noise')
# dfnoise.sort()
global batch_size
batch_size = 100
frame_size = 257
num_hidden = 512
keep_probability = 0.8
num_layers = 2
seq_len = tf.placeholder(tf.int32, None)
keep_pr = tf.placeholder(tf.float32, ())
# q2_x = tf.placeholder(tf.float32, [None, max_length, frame_size]) 
# q2_y = tf.placeholder(tf.float32, [None, max_length, frame_size]) 
# x_abs = tf.placeholder(tf.float32, [None, max_length, frame_size])
# s_abs = tf.placeholder(tf.float32, [None, max_length, frame_size])
q2_x = tf.placeholder(tf.float32, [None, None, frame_size]) 
q2_y = tf.placeholder(tf.float32, [None, None, frame_size]) 
x_abs = tf.placeholder(tf.float32, [None, None, frame_size])
s_abs = tf.placeholder(tf.float32, [None, None, frame_size])
# # #NETWORK

# pdb.set_trace()
def get_a_cell(name, lstm_size, keep_prob):
    name = tf.nn.rnn_cell.LSTMCell(lstm_size)
#    lstm_bwd = tf.nn.rnn_cell.LSTMCell(lstm_size)
    drop = tf.nn.rnn_cell.DropoutWrapper(name, output_keep_prob=keep_probability)
    # drop = tf.nn.dropout(name,rate = 0.2)
    return drop

with tf.name_scope('lstm_fwd'):
     cell_fwd = tf.nn.rnn_cell.MultiRNNCell(
     [get_a_cell('lstm_fwd',num_hidden, keep_probability) for _ in range(num_layers)])
     cell_bwd = tf.nn.rnn_cell.MultiRNNCell(
     [get_a_cell('lstm_bwd',num_hidden, keep_probability) for _ in range(num_layers)]
     )

#with tf.name_scope('lstm_bwd'):
#    cell_bwd = tf.nn.rnn_cell.MultiRNNCell(
#    [get_a_cell('lstm_bwd',num_hidden, keep_probability) for _ in range(num_layers)]
#    )

#
#output,_=tf.nn.dynamic_rnn(cell,q2_x,sequence_length=seq_len,dtype=tf.float32)
output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fwd, cell_bwd , inputs = q2_x, sequence_length=seq_len, dtype = tf.float32)
output = tf.concat(output,2)
rnn_out = tf.layers.dense(output, 257, kernel_initializer=
    tf.contrib.layers.xavier_initializer())
rnn_out = tf.nn.leaky_relu(rnn_out)
# pdb.set_trace()
# fin_out = tf.sigmoid(rnn_out)

dim = seq_len[0]
lr = 0.0001
# cost = tf.reduce_mean(tf.losses.mean_squared_error(rnn_out[:, :dim,:]*x_abs, 
#     s_abs)) + tf.reduce_mean(tf.losses.mean_squared_error(rnn_out[:, :dim,:], 
#     q2_y[:, :dim, :]))
cost=tf.reduce_mean(tf.losses.mean_squared_error(tf.math.log((rnn_out[:, :dim,:])**2), 
    tf.math.log((q2_y[:, :dim, :])**2)))


optimizer = tf.train.AdamOptimizer(learning_rate= lr).minimize(cost)


sess = tf.Session()
saver = tf.train.Saver()


# CODE FOR MODEL COMPRESSION
new_saver=tf.train.Saver(max_to_keep=0)
new_saver.restore(sess,"..Path to the model weights")
# xx = tf.trainable_variables()
tv = tf.trainable_variables()
xx = sess.run(tv)
X = sess.run(tf.expand_dims(tf.concat([tf.reshape(x,[-1]) for x in tf.trainable_variables()], axis = 0), axis = -1))
# len(xx)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=7000, random_state=0).fit(X)
pdb.set_trace()  


sess.close()
