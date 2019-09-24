import numpy as np
import tensorflow as tf


class VoxelAE():


    def __init__(self):
        self.voxel_grid_dims = [32, 32, 32]
        self.latents_num = 100
        self.momentum = 0.9
        self.voxel_channels = 1
        self.batch_size = None

        self.is_train = tf.placeholder(tf.bool, name='holder_is_train')
        self.learning_rate = tf.placeholder(tf.float32, name='holder_learn_rate')
        self.partial_voxel_grids = tf.placeholder(tf.float32, 
                [None, self.voxel_grid_dims[0], self.voxel_grid_dims[1], 
                self.voxel_grid_dims[2], 1], name='partial_voxel_grids')
        self.true_voxel_grids = tf.placeholder(tf.float32, 
                [None, self.voxel_grid_dims[0], self.voxel_grid_dims[1], 
                self.voxel_grid_dims[2], 1], name='true_voxel_grids')

        self.ae_struct_res = {}


    def conv3d(self, x, W, stride, pad):
        # Conv3D wrapper, with elu activation
        x = tf.nn.conv3d(x, W, strides=[1, stride, stride, stride, 1], padding=pad)
        x = tf.layers.batch_normalization(x, training=self.is_train)
        #x = tf.nn.bias_add(x, b)
        #return tf.nn.relu(x)
        return tf.nn.elu(x)


    def conv3d_sigmoid(self, x, W, stride, pad):
        # Conv3D wrapper, with sigmoid activation
        x = tf.nn.conv3d(x, W, strides=[1, stride, stride, stride, 1], padding=pad)
        x = tf.layers.batch_normalization(x, training=self.is_train)
        #x = tf.nn.bias_add(x, b)
        return tf.nn.sigmoid(x)


    def conv3d_transpose(self, x, W, out_shape, stride, pad):
        # Conv3D wrapper, with elu activation
        x = tf.nn.conv3d_transpose(x, W, output_shape=out_shape, 
                                   strides=[1, stride, stride, stride, 1], padding=pad)
        x = tf.layers.batch_normalization(x, training=self.is_train)
        #x = tf.nn.bias_add(x, b)
        #return tf.nn.relu(x)
        return tf.nn.elu(x)

    
    def fully_connected(self, x, W):
        # Fully connected wrapper, with elu activation
        x = tf.matmul(x, W)
        #x = tf.add(tf.matmul(x, W), b)
        x = tf.layers.batch_normalization(x, training=self.is_train)
        #return tf.nn.relu(x)
        return tf.nn.elu(x)
 
    
    def create_ae_enc_var(self, var_scope='voxel_ae_enc_var'):
        # Compute convolution output shape:
        # https://stackoverflow.com/questions/37674306/
        # what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t
        # https://www.tensorflow.org/api_guides/python/nn#Convolution
        self.padding_voxel_enc = {
                'enc_conv1': 'VALID',
                'enc_conv2': 'SAME',
                'enc_conv3': 'VALID',
                'enc_conv4': 'SAME',
                }

        self.strides_voxel_enc = {
                'enc_conv1': 1,
                'enc_conv2': 2,
                'enc_conv3': 1,
                'enc_conv4': 2,
                }

        with tf.variable_scope(var_scope):
            self.weights_voxel_enc = {
                    # Xavier initializion is also called Glorot initialization(tf.glorot_uniform_initializer)
                    'enc_conv1': tf.get_variable(name='w_enc_conv1_voxel', shape=[3, 3, 3, self.voxel_channels, 8],
                            initializer=tf.contrib.layers.xavier_initializer()),
                    'enc_conv2': tf.get_variable(name='w_enc_conv2_voxel', shape=[3, 3, 3, 8, 16],
                            initializer=tf.contrib.layers.xavier_initializer()),
                    'enc_conv3': tf.get_variable(name='w_enc_conv3_voxel', shape=[3, 3, 3, 16, 32],
                            initializer=tf.contrib.layers.xavier_initializer()),
                    'enc_conv4': tf.get_variable(name='w_enc_conv4_voxel', shape=[3, 3, 3, 32, 64],
                            initializer=tf.contrib.layers.xavier_initializer()),
                    'enc_fc1': tf.get_variable(name='w_enc_fc1_voxel', shape=[7*7*7*64, 343], 
                        initializer=tf.contrib.layers.xavier_initializer()),
                    'enc_fc_z': tf.get_variable(name='w_enc_fc_z', shape=[343, self.latents_num], 
                        initializer=tf.contrib.layers.xavier_initializer()),
                    }


    def create_ae_dec_var(self):
        self.padding_voxel_dec = {
                'dec_conv1': 'SAME',
                'dec_conv2': 'VALID',
                'dec_conv3': 'SAME',
                'dec_conv4': 'VALID',
                'dec_conv5': 'SAME',
                }

        self.strides_voxel_dec = {
                'dec_conv1': 1,
                # Transpose convoluation layer
                'dec_conv2': 2,
                'dec_conv3': 1,
                # Transpose convoluation layer
                'dec_conv4': 2,
                'dec_conv5': 1,
                }

        with tf.variable_scope('voxel_ae_dec_var'):
            self.weights_voxel_dec = {
                    'dec_fc1': tf.get_variable(name='w_dec_fc1_voxel', shape=[self.latents_num, 343], 
                        initializer=tf.contrib.layers.xavier_initializer()),
                    'dec_conv1': tf.get_variable(name='w_dec_conv1_voxel', shape=[3, 3, 3, 1, 64],
                            initializer=tf.contrib.layers.xavier_initializer()),
                    'dec_conv2': tf.get_variable(name='w_dec_conv2_voxel', shape=[3, 3, 3, 32, 64],
                            initializer=tf.contrib.layers.xavier_initializer()),
                    'dec_conv3': tf.get_variable(name='w_dec_conv3_voxel', shape=[3, 3, 3, 32, 16],
                            initializer=tf.contrib.layers.xavier_initializer()),
                    'dec_conv4': tf.get_variable(name='w_dec_conv4_voxel', shape=[4, 4, 4, 8, 16],
                            initializer=tf.contrib.layers.xavier_initializer()),
                    'dec_conv5': tf.get_variable(name='w_dec_conv5_voxel', shape=[3, 3, 3, 8, 1],
                            initializer=tf.contrib.layers.xavier_initializer()),
                    }
   

    def build_voxel_enc_struct(self, voxel_grid, w_voxel, strides_voxel, 
                                padding, var_scope='voxel_ae_enc_struct'):
        # Notice: bias is not necessary with batch normalization
        # see the normalizer_fn parameter of:
        # https://www.tensorflow.org/api_docs/python/tf/contrib/layers/fully_connected
        # https://www.tensorflow.org/api_docs/python/tf/contrib/layers/conv3d
        # https://stackoverflow.com/questions/46256747/
        # can-not-use-both-bias-and-batch-normalization-in-convolution-layers
        # https://towardsdatascience.com/batch-normalization-8a2e585775c9
        # Notice: batch normlization first, then call activation function
        # Encode
        with tf.variable_scope(var_scope):
            self.batch_size = tf.shape(voxel_grid)[0]
            enc_conv1 = self.conv3d(voxel_grid, w_voxel['enc_conv1'], strides_voxel['enc_conv1'],
                                    padding['enc_conv1'])
            enc_conv2 = self.conv3d(enc_conv1, w_voxel['enc_conv2'], strides_voxel['enc_conv2'],
                                    padding['enc_conv2'])
            enc_conv3 = self.conv3d(enc_conv2, w_voxel['enc_conv3'], strides_voxel['enc_conv3'],
                                    padding['enc_conv3'])
            enc_conv4 = self.conv3d(enc_conv3, w_voxel['enc_conv4'], strides_voxel['enc_conv4'],
                                    padding['enc_conv4'])
            enc_conv4_flat = tf.reshape(enc_conv4, [-1, int(np.prod(enc_conv4.get_shape()[1:]))])
            enc_fc1 = self.fully_connected(enc_conv4_flat, w_voxel['enc_fc1'])

            enc_z = self.fully_connected(enc_fc1, w_voxel['enc_fc_z'])
        return enc_z


    def build_voxel_dec_struct(self, enc_z, w_voxel, strides_voxel, padding):
            # Decode
            # Transpose convoluation: https://www.matthewzeiler.com/mattzeiler/
            # deconvolutionalnetworks.pdf 
            # How to compute the output shape of tf transpose convolution:
            # https://datascience.stackexchange.com/questions/26451/
            # how-to-calculate-the-output-shape-of-conv2d-transpose
            # Padding==Same:
            # H = H1 * stride
            # Padding==Valid
            # H = (H1-1) * stride + HF
            # where, H = output size, H1 = input size, HF = height of filter
        with tf.variable_scope('voxel_ae_dec_struct'):
            dec_fc1 = self.fully_connected(enc_z, w_voxel['dec_fc1'])
            dec_fc1_unflatten = tf.reshape(dec_fc1, [-1, 7, 7, 7, 1]) 
            dec_conv1 = self.conv3d(dec_fc1_unflatten, w_voxel['dec_conv1'], strides_voxel['dec_conv1'],
                                    padding['dec_conv1'])
            dconv2_output_shape = [self.batch_size, 15, 15, 15, 32]
            dec_conv2 = self.conv3d_transpose(dec_conv1, w_voxel['dec_conv2'], dconv2_output_shape,
                                              strides_voxel['dec_conv2'], padding['dec_conv2'])
            dec_conv3 = self.conv3d(dec_conv2, w_voxel['dec_conv3'], strides_voxel['dec_conv3'],
                                                padding['dec_conv3'])
            dconv4_output_shape = [self.batch_size, 32, 32, 32, 8]
            dec_conv4 = self.conv3d_transpose(dec_conv3, w_voxel['dec_conv4'], dconv4_output_shape,
                                              strides_voxel['dec_conv4'], padding['dec_conv4'])
            dec_conv5 = self.conv3d_sigmoid(dec_conv4, w_voxel['dec_conv5'], strides_voxel['dec_conv5'],
                                                padding['dec_conv5'])
            #dec_conv5 = self.conv3d(dec_conv4, w_voxel['dec_conv5'], strides_voxel['dec_conv5'],
            #                                    padding['dec_conv5'])
            #dec_conv5 = tf.nn.sigmoid(dec_conv5)
            #dec_conv5 = 0.1 + 0.9 * dec_conv5
            return dec_conv5


    # Weighted binary cross-entropy for use in voxel loss. 
    # Allows weighting of false positives relative to false negatives.
    # Nominally set to strongly penalize false negatives
    #def weighted_binary_crossentropy(self, output, target):
    #    return -(98.0*target * tf.log(output) + 2.0*(1.0 - target) * tf.log(1.0 - output))/100.0


    def binary_crossentropy(self, output, target):
        return -(target * tf.log(output) + (1.0 - target) * tf.log(1.0 - output))


    def ae_loss(self, voxel, voxel_reconstructed, output_layer_w, train_mode=True):
        # Notice: The loss function of the variational autoencoder 
        # is the negative log-likelihood with a regularizer.        
        # Voxel-Wise Reconstruction Loss 
        # Note that the output values are clipped to prevent the BCE from evaluating log(0).
        with tf.variable_scope('voxel_ae_loss'):
            voxel_loss = self.binary_crossentropy(tf.clip_by_value(voxel_reconstructed, 
                                                            1e-7, 1.0 - 1e-7), voxel)
            #voxel_loss = self.weighted_binary_crossentropy(tf.clip_by_value(voxel_reconstructed, 
            #                                                0.5, 1.0 - 1e-7), voxel)
            voxel_loss = tf.reduce_sum(voxel_loss, 1)
            voxel_loss = tf.reduce_mean(voxel_loss)
            # L2 normalization for the output layer
            l2_loss = tf.nn.l2_loss(output_layer_w)
            loss = voxel_loss + l2_loss
            if train_mode:
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    #optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
                    optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, 
                                                           momentum=self.momentum,
                                                           use_nesterov=True).minimize(loss)
        voxel_loss_sum = tf.summary.scalar('voxel_loss', voxel_loss)
        l2_loss_sum = tf.summary.scalar('l2_loss', l2_loss)
        loss_sum = tf.summary.scalar('loss', loss)
        learn_rate_sum = tf.summary.scalar('learning_rate', self.learning_rate)
        train_summary = tf.summary.merge([voxel_loss_sum, l2_loss_sum, 
                                          loss_sum, learn_rate_sum])
        res = {'train_summary': train_summary,
                'loss': loss,
               }
        if train_mode:
            res['optimizer'] = optimizer
        return res


    def build_voxel_ae_enc(self):
        self.create_ae_enc_var()
        enc_z = self.build_voxel_enc_struct(self.partial_voxel_grids, self.weights_voxel_enc, 
                                    self.strides_voxel_enc, self.padding_voxel_enc)
        self.ae_struct_res['embedding'] = enc_z


    def build_voxel_ae_dec(self):
        self.create_ae_dec_var()
        dec_conv5 = self.build_voxel_dec_struct(self.ae_struct_res['embedding'], 
                                    self.weights_voxel_dec, self.strides_voxel_dec,
                                    self.padding_voxel_dec)
        self.ae_struct_res['voxel_reconstructed'] = dec_conv5


    def build_voxel_ae(self):
        self.build_voxel_ae_enc()
        self.build_voxel_ae_dec()


    def train_voxel_ae_model(self):
        self.build_voxel_ae()
        self.voxel_reconstructed = self.ae_struct_res['voxel_reconstructed']
        ae_loss_res = self.ae_loss(self.true_voxel_grids, self.ae_struct_res['voxel_reconstructed'], 
                                    self.weights_voxel_dec['dec_conv5'])
        self.optimizer = ae_loss_res['optimizer']
        self.train_summary = ae_loss_res['train_summary']
        self.loss = ae_loss_res['loss']


    def test_voxel_ae_model(self):
        self.build_voxel_ae()
        self.voxel_reconstructed = self.ae_struct_res['voxel_reconstructed']
        self.embedding = self.ae_struct_res['embedding']
        ae_loss_res = self.ae_loss(self.true_voxel_grids, self.ae_struct_res['voxel_reconstructed'], 
                                    self.weights_voxel_dec['dec_conv5'], train_mode=False)
        self.loss = ae_loss_res['loss']

