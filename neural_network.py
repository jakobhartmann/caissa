import requests
import json
import grpc

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, BatchNormalization, ReLU, Add
from tensorflow.keras.backend import clip, epsilon, log, sum, square, mean
from tensorflow.keras import regularizers
from tensorflow.core.framework import types_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

# from tic_tac_toe.config import Config

from three_check_chess.config import Config


'''Contains architecture and losses of the neural network together with functions to make inference requests via the REST- and gRPC-API to TensorFlow Serving'''
class NeuralNetwork:
    '''Initialize the class variables'''
    def __init__(self):
        # Name of the model
        self.model_name = Config.MODEL_NAME
        # Shape of the neural network input, i.e., the bitboard representation
        self.input_shape = Config.INPUT_SHAPE
        # Number of residual blocks in the body of the neural network
        self.num_residual_blocks = Config.NUM_RESIDUAL_BLOCKS
        # Number of convolutional filters in the body and first layer of the policy head
        self.num_convolutional_filters = Config.NUM_CONVOLUTIONAL_FILTERS
        # Number of convolutional filters in the policy head, representing the number of output planes
        self.num_policy_head_filters = Config.NUM_POLICY_HEAD_FILTERS
        # Number of convolutional filters in the value head
        self.num_value_head_filters = Config.NUM_VALUE_HEAD_FILTERS
        # Number of possible legal and illegal moves that the neural network can represent
        self.num_moves = Config.NUM_MOVES
        # L2 regularization strength for the kernel and bias regularizer in the convoluational and dense layers
        self.l2_regularization_strength = Config.L2_REGULARIZATION_STRENGTH
        # TensorFlow Serving URL for inference requests
        self.model_url = Config.MODEL_URL
        # TenorFlow Serving REST port for inference requests 
        self.rest_port = Config.REST_PORT
        # TenorFlow Serving gRPC port for inference requests
        self.grpc_port = Config.GRPC_PORT


    '''Returns a residual block'''
    def get_residual_block(self, input):
        x = Conv2D(filters = self.num_convolutional_filters, kernel_size = (3, 3), strides = (1, 1), padding = 'same', kernel_regularizer = regularizers.l2(self.l2_regularization_strength), bias_regularizer = regularizers.l2(self.l2_regularization_strength))(input)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters = self.num_convolutional_filters, kernel_size = (3, 3), strides = (1, 1), padding = 'same', kernel_regularizer = regularizers.l2(self.l2_regularization_strength), bias_regularizer = regularizers.l2(self.l2_regularization_strength))(x)
        x = BatchNormalization()(x)
        x = Add()([input, x])
        x = ReLU()(x)
        return x


    '''Returns a convolutional recurrent neural network with a policy and value head and their shared body consisting of several residual blocks'''
    def get_model(self):
        # Input
        inputs = Input(shape = self.input_shape)

        # Body
        body = Conv2D(filters = self.num_convolutional_filters, kernel_size = (3, 3), strides = (1, 1), padding = 'same', kernel_regularizer = regularizers.l2(self.l2_regularization_strength), bias_regularizer = regularizers.l2(self.l2_regularization_strength))(inputs)
        body = BatchNormalization()(body)
        body = ReLU()(body)

        # Residual blocks
        for _ in range(self.num_residual_blocks):
            body = self.get_residual_block(body)

        # Policy head
        policy_head = Conv2D(filters = self.num_convolutional_filters, kernel_size = (3, 3), strides = (1, 1), padding = 'same', kernel_regularizer = regularizers.l2(self.l2_regularization_strength), bias_regularizer = regularizers.l2(self.l2_regularization_strength))(body)
        policy_head = BatchNormalization()(policy_head)
        policy_head = ReLU()(policy_head)
        policy_head = Conv2D(filters = self.num_policy_head_filters, kernel_size = (3, 3), strides = (1, 1), padding = 'same', kernel_regularizer = regularizers.l2(self.l2_regularization_strength), bias_regularizer = regularizers.l2(self.l2_regularization_strength))(policy_head)
        policy_head = Flatten()(policy_head)
        policy_head = tf.keras.activations.softmax(policy_head)

        # Value head
        value_head = Conv2D(filters = self.num_value_head_filters, kernel_size = (1, 1), strides = (1, 1), padding = 'same', kernel_regularizer = regularizers.l2(self.l2_regularization_strength), bias_regularizer = regularizers.l2(self.l2_regularization_strength))(body)
        value_head = BatchNormalization()(value_head)
        value_head = ReLU()(value_head)
        value_head = Flatten()(value_head)
        value_head = Dense(units = 256)(value_head)
        value_head = ReLU()(value_head)
        value_head = Dense(units = 1, activation = 'tanh', kernel_regularizer = regularizers.l2(self.l2_regularization_strength), bias_regularizer = regularizers.l2(self.l2_regularization_strength))(value_head)

        # Build Keras model
        model = Model(inputs = inputs, outputs = [policy_head, value_head], name = self.model_name)

        # Return model
        return model


    '''Makes a request to the neural network of the given bitboard using the REST API'''
    def make_rest_prediction(self, bitboard):
        url = 'http://' + self.model_url + ':' + str(Config.REST_PORT) + '/v1/models/' + self.model_name + ':predict'
        data = json.dumps({"signature_name": "serving_default", "instances": np.array(bitboard).tolist()})
        headers = {"content-type": "application/json"}
        json_response = requests.post(url, data = data, headers = headers)
        predictions = json.loads(json_response.text)['predictions'][0]
    
        # policy = predictions['tf.compat.v1.nn.softmax']
        policy = predictions['tf.nn.softmax']
        value = predictions['dense_1'][0]
        
        return [policy], value


    '''Makes a request to the neural network of the given bitboard using the gRPC API'''
    def make_grpc_prediction(self, bitboard):
        channel = grpc.insecure_channel(self.model_url + ':' + str(self.grpc_port))
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        grpc_request = predict_pb2.PredictRequest()
        grpc_request.model_spec.name = self.model_name
        grpc_request.model_spec.signature_name = 'serving_default'
        grpc_request.inputs['input_1'].CopyFrom(tf.make_tensor_proto(bitboard, dtype = types_pb2.DT_FLOAT))
        result = stub.Predict(grpc_request)

        # policy = result.outputs['flatten'].float_val
        # policy = result.outputs['tf.compat.v1.nn.softmax'].float_val
        policy = result.outputs['tf.nn.softmax'].float_val
        value = result.outputs['dense_1'].float_val[0]
        
        return [policy], value


    '''Returns the mean squared error between the given inputs'''
    def mse_loss(self, y_true, y_pred):
        return mean(square(y_pred - y_true))


    '''Returns the cross entropy between the given inputs'''
    def ce_loss(self, y_true, y_pred):
        y_true = clip(y_true, epsilon(), 1)
        y_pred = clip(y_pred, epsilon(), 1)
        return -sum(y_true * log(y_pred), axis = -1)