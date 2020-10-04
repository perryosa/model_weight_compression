# model_weight_compression

As asked to show one of the method for model compression. I am providing the code for the same in model_compression.py file.

Navigation for the Code :-

1.Given architecture in the code is my own archtecture for speech enhancement. [Paper has been accepter for publication in IEEE Signal Processing Letters] So I have no prolem in sharing the code.
<br />
2.Weights that I have used for showing the weight compression is also my model weights trained on mozilla speech corpus.
\n
3. After loading the model and model weights model compression code starts.


NOTE- As for reference I have not taken help of any website or article. Upon reading the problem statement I have given it a big thought to how we can tackle this problemt


METHOD:-
I have extracted the weights of all the model layers. Reshaped the weights of individual layer to flatten(convert into 1-D).
After flattening the individual layer weights I have concatenated all the layer weights together to make it a big 1-D weight array.
The size initially was around (700000 , 1)
After that I have applied KMeans clustering(I have written the code for it initialy but it was slow so I have used sklearn) on the 1-D array.
In KMeans I have choosed the cluster size to be 7000. So all the 1-D 7lack points are assigned to only 7000 clusters. In other words our 7 lack weights that we needed to store are compressed to 7 thousand weights. Using this we eed around 100 times lower space for saving weights.
Plus we need one look-up table that we need to decompress the weights at the time of computation. That look-up table will be from Kmeans clustering that will give us the index of weight and corresponding cluster value.

Using this method we can compress our model according to number of clusters. As the number of clusters decreases accuracy will decrese. Sp we will experimentally find a trade-off value between size of modek weights(no of cluster) and accuracy.

Disclaimer - Code will take a lot of time to calculate 7000 clusters. So Please be patient. (Around 15 mins in my laptop)
