# Code adapté de projets académiques de la professeur Fei Fei Li et de ses étudiants Andrej Karpathy, Justin Johnson et autres.
# Version finale rédigée par Carl Lemaire, Vincent Ducharme et Pierre-Marc Jodoin

import numpy as np

from ift725.layers import *
from ift725.quick_layers import *
from ift725.layer_combo import *


class ThreeLayerConvolutionalNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-2, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialisez les poids et les biais pour le réseau de convolution à #
        #  trois couches.                                                          #
        # Les poids devraient être initialisé à partir d'une Gaussienne avec un    #
        # écart-type égal à weight_scale; les biais devraient être initialisés à 0.#
        # Tous les poids et les biais devraient être emmagasinés dans le           #
        # dictionnaire self.params.                                                #
        # Emmagasinez les poids et les biais de la couche de convolution en        #
        # utilisant les clés 'W1' et 'b1' respectivement; utilisez les clés 'W2'   #
        # et 'b2' pour les poids et les biais de la couche cachée affine et        #
        # utilisez les clés 'W3' et 'b3' pour les poids et les biais de la couche  #
        # affine de sortie.                                                        #
        ############################################################################

        C, H, W = input_dim

        self.params['W1'] = np.random.randn(num_filters, C, filter_size, filter_size) * weight_scale
        self.params['W2'] = np.random.randn(num_filters * int((H / 2)) * int((W / 2)), hidden_dim) * weight_scale
        self.params['W3'] = np.random.randn(hidden_dim, num_classes) * weight_scale

        self.params['b1'] = np.zeros(num_filters)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['b3'] = np.zeros(num_classes)

        ############################################################################
        #                             FIN DE VOTRE CODE                            #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Inputs:
        - X: Array of input data of shape (N, C, H, W)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implémentez la propagation pour ce réseau de convolution à trois   #
        #  couches, calculant les scores de classes pour X et stockez-les dans la  #
        #  variable scores.                                                        #
        ############################################################################

        out, conv_cache = forward_convolutional_relu_pool(X, W1, b1, conv_param, pool_param)
        out, relu_cache = forward_fully_connected_transform_relu(out, W2, b2)
        scores, score_cache = forward_fully_connected(out, W3, b3)

        ############################################################################
        #                             FIN DE VOTRE CODE                            #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implémentez la passe arrière pour ce réseau de convolution à trois #
        #  couches, en stockant la perte et les gradients dans les variables loss  #
        #  et grads.                                                               #
        # Calculez la perte de données en utilisant softmax et assurez-vous que    #
        # grads[k] contient les gradients pour self.params[k]. N'oubliez pas       #
        # d'ajouter la régularisation L2!                                          #
        ############################################################################

        loss, dx = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))

        dx3, dW3, db3 = backward_fully_connected(dx, score_cache)
        dx2, dW2, db2 = backward_fully_connected_transform_relu(dx3, relu_cache)
        dx, dW1, db1 = backward_convolutional_relu_pool(dx2, conv_cache)

        grads['W3'] = dW3 + self.reg * W3
        grads['W2'] = dW2 + self.reg * W2
        grads['W1'] = dW1 + self.reg * W1

        grads['b3'] = db3 + self.reg * db3
        grads['b2'] = db2 + self.reg * db2
        grads['b1'] = db1 + self.reg * db1

        ############################################################################
        #                             FIN DE VOTRE CODE                            #
        ############################################################################

        return loss, grads


pass
