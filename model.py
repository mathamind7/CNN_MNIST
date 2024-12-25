import numpy as np
from scipy import signal 

class Convolutional:
    def __init__(self,input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_depth = input_depth
        self.input_shape = input_shape
        self.kernal_shape = (depth, input_depth, kernel_size, kernel_size)
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        
        self.kernels = np.random.randn(*self.kernal_shape)
        self.bias = np.random.randn(*self.output_shape)
        
    def foward(self,input_data):
        self.input_data = input_data
        self.output = np.copy(self.bias)
        
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input_data[j], self.kernels[i, j], "valid")
        
        return self.output 
    
    def backward(self, output_gradient, learning_rate):
        kernel_gradient = np.zeros(self.kernal_shape)
        input_gradient  = np.zeros(self.input_shape)
        
        for i in range(self.depth):
            for j in range(self.input_depth):
                kernel_gradient[i, j] = signal.correlate2d(self.input_data[j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[j],self.kernels[i, j], "full")
        
        self.kernels -= learning_rate * kernel_gradient
        self.bias -= learning_rate * output_gradient
        
        return input_gradient