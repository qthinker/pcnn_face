# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 21:03:06 2014

@author: break
"""
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
from dataset import loadFaceVerifyDataSet
import gzip
import cPickle



class MyConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input1, input2, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a MyConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input1: theano.tensor.dtensor4
        :param input1: symbolic image tensor, of shape image_shape, the one image of a pair
        
        :type input2: theano.tensor.dtensor4
        :param input2: symbolic image tensor, of shape image_shape, the other image of a pair

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (,num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input1 = input1
        self.input2 = input2

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                   np.prod(poolsize))
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(np.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out1 = conv.conv2d(input=input1, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape)
        conv_out2 = conv.conv2d(input=input2, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape)

        # downsample each feature map individually, using maxpooling
        pooled_out1 = downsample.max_pool_2d(input=conv_out1,
                                            ds=poolsize, ignore_border=True)
        pooled_out2 = downsample.max_pool_2d(input=conv_out2,
                                            ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output1 = T.tanh(pooled_out1 + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output2 = T.tanh(pooled_out2 + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]
        out_h = (image_shape[2] - filter_shape[2] + 1) / poolsize[0]
        out_w = (image_shape[3] - filter_shape[3] + 1) / poolsize[1]
        self.out_shape = (1,filter_shape[0], out_h, out_w)

class HiddenLayer(object):
    def __init__(self, rng, input1, input2, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input1: theano.tensor.dmatrix
        :param input1: a symbolic tensor of shape ( n_in)
        
        :type input2: theano.tensor.dmatrix
        :param input2: a symbolic tensor of shape ( n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input1 = input1
        self.input2 = input2

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = np.asarray(rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output1 = T.dot(input1, self.W) + self.b
        self.output1 = (lin_output1 if activation is None
                       else activation(lin_output1))
        lin_output2 = T.dot(input2, self.W) + self.b
        self.output2 = (lin_output2 if activation is None
                       else activation(lin_output2))
        # parameters of the model
        self.params = [self.W, self.b]
    def loss(self,delta):
        #return T.log(1+T.exp(euclid(self.output1,self.output2)))
        #return T.log(1+T.exp(T.sqrt(T.sum(T.sqr(self.output1-self.output2)))))
        #return T.log(1+T.exp(T.sqrt(T.sum(T.sqr(self.output1)))))
        return T.log(1+T.exp(delta*(T.sum(T.sqr(self.output1-self.output2)))))
    
def euclid(I_1, I_2):
    """
    compute euclid distance of two vectors
    """
    return T.sqrt(T.sum(T.sqr(I_1-I_2)))
        
def faceRecognition(learning_rate=0.1, n_epochs=10,
                    dataset='face_data_pcnn.pkl.gz',
                    nkerns=[10, 20], outDims=[324, 98]):
    """
    Face recognition with Pyramid CNN architecture
    
    """
    
    assert len(nkerns) == len(outDims)
    levels = len(nkerns)
    rng = np.random.RandomState(12345)
    
    data_x, data_y = loadFaceVerifyDataSet(dataset)
    data_length = len(data_x.get_value(borrow=True))
    print "data_length:%d" % data_length
    
    
    # allocate symbolic variables for the data
    index_i = T.iscalar()  
    
    x1 = T.vector('x1')   # 
    x2 = T.vector('x2')     #image pair
    
    delta = T.iscalar('delta')  #label
    
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    
    layer0_input1 = x1.reshape((1, 1, 40, 40))
    layer0_input2 = x2.reshape((1, 1, 40, 40))
    
    level_cnn = []
    level_mlp = []
    
    for i in range(levels):
        print "...building net level %d\n" %i
        if(i == 0):
            cnn_i = MyConvPoolLayer(rng, input1=layer0_input1, 
                                    input2=layer0_input2,
                                    image_shape=(1, 1,40,40),
                                    filter_shape=(nkerns[0],1,5,5),
                                    poolsize=(2,2))
            level_cnn.append(cnn_i)
            mlp_i = HiddenLayer(rng, input1=cnn_i.output1.flatten(),
                                input2=cnn_i.output2.flatten(),
                                n_in=np.prod(cnn_i.out_shape),
                                n_out=outDims[0])
            level_mlp.append(mlp_i)
        else:
            cnn_i = MyConvPoolLayer(rng, input1=level_cnn[i-1].output1,
                                    input2=level_cnn[i-1].output2,
                                    image_shape=level_cnn[i-1].out_shape,
                                    filter_shape=(nkerns[i],nkerns[i-1],5,5),
                                    poolsize=(2,2))
            level_cnn.append(cnn_i)
            mlp_i = HiddenLayer(rng, input1=cnn_i.output1.flatten(),
                                input2=cnn_i.output2.flatten(),
                                n_in=np.prod(cnn_i.out_shape),
                                n_out=outDims[i])
            level_mlp.append(mlp_i)
    
    train = []
    for i in range(levels):
        print "...building update level %d\n" % i
        #delta = T.iscalar()
        
        L = level_mlp[i].loss(delta)
        params = level_cnn[i].params + level_mlp[i].params
        grads = T.grad(L, params)
        
        updates = []

        for param_i, grad_i in zip(params, grads):
            updates.append((param_i, param_i - learning_rate * grad_i))
      
        train_model = theano.function([index_i], L, updates=updates,
                                  givens={
                                  x1:data_x[index_i][0],
                                  x2:data_x[index_i][1],
                                  delta:data_y[index_i]},
                                  on_unused_input='ignore'
                                  )
        train.append(train_model)
    
    print "...build model compete"
    fr = open('pcnn_result_new.txt','wt')
    for epoch in range(n_epochs):
        for i in range(data_length):
            for k in range(levels):
                L = train[k](i)
                print "epoch:%d\ti=%d\tk=%d\tL=%s\n" % (epoch,i,k,L)
                #print L
                fr.writelines("epoch:%d\ti=%d\tk=%d\tL=%s\n" % (epoch,i,k,L))
    fr.close()
    #save the params
    f=gzip.open('pcnn_params.pkl.gz','wb')
    cPickle.dump(level_cnn[0].params+level_mlp[0].params+level_cnn[1].params+level_mlp[1].params,f)
    f.close()    
if __name__ == "__main__":
    #faceRecognition()
    f=gzip.open('pcnn_params.pkl.gz','rb')
    params = cPickle.load(f)
    f.close()
    print(params[5].get_value())
    