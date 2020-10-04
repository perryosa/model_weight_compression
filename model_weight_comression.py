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
new_saver.restore(sess,'/home/perrryosa/Speech-Denoising-With-RNN/dataset_generate/wind_dataset_IITGN/pink_wind_dataset/test_pink_wind/model/bi_LSTM_stft_pink_wind_IITGN')
# xx = tf.trainable_variables()
tv = tf.trainable_variables()
xx = sess.run(tv)
X = sess.run(tf.expand_dims(tf.concat([tf.reshape(x,[-1]) for x in tf.trainable_variables()], axis = 0), axis = -1))
# len(xx)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=50, random_state=0).fit(X)
pdb.set_trace()  


with tf.Session() as sess:
# new_saver = tf.train.import_meta_graph("/home/perrryosa/Speech-Denoising-With-RNN/q2model/my_rnn_model.meta")
    new_saver=tf.train.Saver(max_to_keep=0)
    # new_saver.restore(sess, tf.train.latest_checkpoint('./'))

    # new_saver.restore(sess,tf.train.latest_checkpoint('/home/perrryosa/Speech-Denoising-With-RNN/result multilayer/q2model/'))

    new_saver.restore(sess,'/home/perrryosa/Speech-Denoising-With-RNN/dataset_generate/wind_dataset_IITGN/pink_wind_dataset/test_pink_wind/model/bi_LSTM_stft_pink_wind_IITGN')
    print(tf.trainable_variables())
    pdb.set_trace()
    # print_tensors_in_checkpoint_file("/home/perrryosa/Speech-Denoising-With-RNN/q2model/new_data_apple/combined_loss/CRNN_stft", all_tensors=False, tensor_name='')
    # with tf.variable_scope("rnn_out", reuse=True):
    #     w = tf.get_variable("kernel")
    #     tf.Print(w, [w])
        # Also tried tf.Print(hidden_layer2, [w])

    # pdb.set_trace()
    # a = np.arange(0,100)
    # b=[]
    # for i in range(len(a)):
    #     b.append(str(a[i]))
    # b.sort()
    # # pdb.set_trace()

    # # SNR = ['0', '10', '5', 'm5']
    # data_path =  '/home/perrryosa/Speech-Denoising-With-RNN/dataset_generate/wind_dataset_IITGN/pink_wind_dataset/test_pink_wind' 
    # #dftest=os.listdir('/home/perrryosa/Speech-Denoising-With-RNN/new_data_apple/validation')
    # dftest=os.listdir('/home/perrryosa/Speech-Denoising-With-RNN/dataset_generate/wind_dataset_IITGN/pink_wind_dataset/test_pink_wind/noisy')
    # dftest.sort()
    # tex, TEX, TEX_abs, TEX_len,phase = loadfile(data_path,dftest, 'noisy', flag = 1)
    # print(len(phase), phase[0].shape)
    # def test_SNR(M_pred, X, i):
    #     S_pred = M_pred.T
    #     s_pred = librosa.istft(S_pred, win_length = 512, hop_length = 257)
    #     print('writing audios...')
    #     # librosa.output.write_wav('/home/perrryosa/Speech-Denoising-With-RNN/QANTAS-1-Channel/recovered/recovered'
    #     #  + str(i) + '.wav', s_pred, 16000)
    #     print(s_pred.shape)
    #     wavfile.write('/home/perrryosa/Speech-Denoising-With-RNN/dataset_generate/wind_dataset_IITGN/pink_wind_dataset/test_pink_wind/denoised/recovered_S_' 
    #      + b[i] + '.wav', 16000, s_pred)
    #     # librosa.output.write_wav('/home/perrryosa/Speech-Denoising-With-RNN/test_result/denoised'+str(i)+'.wav', s_pred, 16000)


    # #Getting predictions for all test sets
    # for i in range(len(TEX_abs)):
    #     print(i)
    #     epoch_x = np.zeros((1, TEX_abs[i].shape[1], TEX_abs[i].shape[0]))
    #     epoch_y = np.zeros((1, TEX_abs[i].shape[1], TEX_abs[i].shape[0]))
    #     epoch_x[0,:,:] = TEX_abs[i].T
    #     # print(epoch_x)
    #     epoch_x = np.reshape(epoch_x,(1,TEX_len[i],257))
    #     TEM_pred= sess.run(rnn_out, feed_dict = {q2_x:epoch_x, seq_len : [TEX_len[i]] ,keep_pr: 1})
    #     phase[i] = phase[i].T
    #     phase[i] = np.reshape(phase[i],(1,TEX_len[i],257))
    #     TEM_pred = TEM_pred*phase[i]
    #     print(phase[i].shape,TEM_pred.shape)
    #     test_SNR(TEM_pred[0,:TEX_len[i],:], TEX[i], i)


sess.close()