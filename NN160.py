# import numpy as np
import cupy as cp
import cupyx

rng = cp.random.default_rng(1)

def load_products():
    print("\n*********  NORMAL MULTIPLICATION METHOD  [FP16] *******\n")

load_products()



def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = cp.repeat(cp.arange(field_height), field_width)
    i0 = cp.tile(i0, C)
    i1 = stride * cp.repeat(cp.arange(out_height), out_width)
    j0 = cp.tile(cp.arange(field_width), field_height * C)
    j1 = stride * cp.tile(cp.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = cp.repeat(cp.arange(C), field_height * field_width).reshape(-1, 1)

    return (k.astype(int), i.astype(int), j.astype(int))


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = cp.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols.astype(cp.float16)


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at or  cupyx.scatter_add """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = cp.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    cupyx.scatter_add(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return (x_padded[:, :, padding:-padding, padding:-padding]).astype(cp.float16)







###########################################################################################
###########################################################################################

def idot(A, W, Out):
    cp.dot(A, W, out=Out)


def iHadm(A, B, Out):
    cp.multiply(A, B, out=Out)


def iHads(scalar, A, Out):
    cp.multiply(scalar, A, out=Out)
    return Out


##################################################################################################
####################################### nn modules:###############################################

class Linear:
    def __init__(self, BatchSize: int, InputSize: int, OutputSize: int):
        """
        X or A (BatchSize x InputSize)
        W (InputSize x OutputSize)
        Y (BatchSize x OutputSize)
        """
        self.BatchSize = BatchSize
        self.InputSize = InputSize
        self.OutputSize = OutputSize
        self.X = cp.zeros((BatchSize, InputSize), dtype=cp.float16)  # null
        self.Y = cp.zeros((BatchSize, OutputSize), dtype=cp.float16)

        self.W = (rng.standard_normal(size=(InputSize, OutputSize), dtype=cp.float32) / 
            cp.sqrt(InputSize / 2)
        ).astype(cp.float16)
        self.B = cp.zeros((1, OutputSize), dtype=cp.float16)

        self.dX = cp.zeros((BatchSize, InputSize), dtype=cp.float16)
        self.dW = cp.zeros((InputSize, OutputSize), dtype=cp.float16)
        self.dB = cp.zeros((1, OutputSize), dtype=cp.float16)

    def Forward(self, preX):
        self.X = preX
        idot(self.X, self.W, self.Y)
        self.Y += self.B
        return self.Y

    def Backward(self, dOut):
        idot(self.X.T.copy(), dOut, self.dW)
        idot(dOut, self.W.T.copy(), self.dX)
        cp.sum(dOut, axis=0, out=self.dB, keepdims=True)
        return self.dX

    def Update(self, lr=0.001):
        self.W -= iHads(cp.float16(lr), self.dW, self.dW)
        self.B -= iHads(cp.float16(lr), self.dB, self.dB)
        
        

        

class Conv:
    def __init__(self, Xshape, Wshape, stride=1, padding=1):
        """
        Xshape = (n_x, d_x, h_x, w_x)
        Wshape = (number of filters, Channel of filter, h_filter, w_filter)
        """
        if len(Xshape) != 4:
             raise Exception('Invalid Xshape dimension!')
        if len(Wshape) != 4:
             raise Exception('Invalid Wshape dimension!')
        self.Wshape = Wshape
        self.Xshape = Xshape
        
        self.stride = stride
        self.padding = padding
        
        self.n_filters, self.d_filter, self.h_filter, self.w_filter = self.Wshape
        self.n_x, self.d_x, self.h_x, self.w_x = self.Xshape
        h_out = (self.h_x - self.h_filter + 2 * self.padding) / self.stride + 1
        w_out = (self.w_x - self.w_filter + 2 * self.padding) / self.stride + 1

        if not h_out.is_integer() or not w_out.is_integer():
            raise Exception('Invalid output dimension!')

        self.h_out, self.w_out = int(h_out), int(w_out)
        
        #######
        
        self.W = (rng.standard_normal(size=(self.n_filters, self.d_filter, self.h_filter, self.w_filter), dtype=cp.float32) / 
            cp.sqrt(self.n_filters / 2.0)
        ).astype(cp.float16)
        
        self.B = cp.zeros((self.n_filters,1), dtype=cp.float16)
        self.Y = cp.zeros((self.n_x,self.n_filters,self.h_out, self.w_out), dtype=cp.float16)
        
        self.Wcol = cp.array([0], dtype=cp.float16)  # null
        
        self.Xcol = cp.zeros((self.n_filters, self.d_filter, self.h_filter, self.w_filter, self.n_x, self.h_out, self.w_out), dtype=cp.float16)
        
        self.dXcol = cp.zeros((self.d_filter, self.h_filter, self.w_filter, self.n_x, self.h_out, self.w_out), dtype=cp.float16)
        
        self.dX = cp.zeros(self.Xshape, dtype=cp.float16)
        self.dW = cp.zeros(self.Wshape, dtype=cp.float16)
        self.dB = cp.zeros((self.n_filters,1), dtype=cp.float16)
    
    def Forward(self, preX):
        self.Xcol = im2col_indices(preX, self.h_filter, self.w_filter, padding=self.padding, stride=self.stride)
        self.Wcol = self.W.reshape(self.n_filters, -1).copy()

        idot(self.Wcol,self.Xcol, self.Y)
        self.Y = self.Y.reshape(self.n_filters, -1)
        self.Y += self.B
        self.Y = self.Y.reshape(self.n_filters, self.h_out, self.w_out, self.n_x)
        self.Y = self.Y.transpose(3, 0, 1, 2).copy()
        return self.Y
    
    def Backward(self, dOut):
        self.dB = self.dB.reshape(self.n_filters)
        cp.sum(dOut, axis=(0, 2, 3), out=self.dB)
        self.dB = self.dB.reshape(self.n_filters,-1)
        
        dout_reshaped = dOut.transpose(1, 2, 3, 0).reshape(self.n_filters, -1).copy()
        idot(dout_reshaped, self.Xcol.T.copy(), self.dW)
        self.dW = self.dW.reshape(self.Wshape).copy()
        
        W_reshape = self.W.reshape(self.n_filters, -1)
        
        idot(W_reshape.T.copy(),dout_reshaped, self.dXcol)
        self.dX = col2im_indices(self.dXcol, self.Xshape, self.h_filter, self.w_filter, padding=self.padding, stride=self.stride)
        return self.dX
    
    def Update(self, lr=0.001):
        self.dW = self.dW.reshape(self.n_filters,-1)
        iHads(cp.float16(lr), self.dW, self.dW)
        self.W -= self.dW.reshape(self.Wshape)
        self.B -= iHads(cp.float16(lr), self.dB, self.dB)
        # clear dW, dB, dX
        
class Sigmoid:
    def __init__(self, BatchSize: int, OutputSize: int):
        self.Out = cp.array([0], dtype=cp.float16)
        self.dX = cp.zeros((BatchSize, OutputSize), dtype=cp.float16)

    def Forward(self, X):
        self.Out = (-1 * X).copy()
        cp.exp(self.Out, out=self.Out, dtype=cp.float16)
        cp.divide(1.0, (1.0 + self.Out), out=self.Out, dtype=cp.float16)
        return self.Out

    def Backward(self, dOut):
        iHadm(self.Out, (1.0 - self.Out), self.dX)
        iHadm(self.dX, dOut, self.dX)
        return self.dX

    def Update(self, lr=0.001):
        return lr

class Reshape:
    def __init__(self, Inshape, Outshape):
        self.InSh = Inshape
        self.OutSh = Outshape
    
    def Forward(self,X):
        return X.reshape(self.OutSh).copy()
    
    def Backward(self,dX):
        return dX.reshape(self.InSh).copy()
    
    def Update(self, lr=0.001):
        return lr
    

class Relu:
    def __init__(self):
        self.In = cp.array([0], dtype=cp.float16)
        self.Out = cp.array([0], dtype=cp.float16)
        self.dX = cp.array([0], dtype=cp.float16)

    def Forward(self, X):
        self.In = X.copy()
        self.Out = X
        self.Out[self.In < 0] = 0
        return self.Out

    def Backward(self, dOut):
        self.dX = dOut
        self.dX[self.In <= 0] = 0
        return self.dX

    def Update(self, lr=0.001):
        return lr

class LRelu:
    def __init__(self, leak= 0.01):
        self.In = cp.array([0], dtype=cp.float16)
        self.Out = cp.array([0], dtype=cp.float16)
        self.dX = cp.array([0], dtype=cp.float16)
        self.leak = cp.float16(leak)

    def Forward(self, X):
        self.In = X.copy()
        self.Out = X
        self.Out[self.In < 0] *= self.leak
        return self.Out

    def Backward(self, dOut):
        self.dX = dOut
        self.dX[self.In <= 0] *= self.leak
        return self.dX

    def Update(self, lr=0.001):
        return lr

    
    
    
    
###################################################################

def OneHot(labels, Category):
    y = cp.zeros((labels.size, Category), dtype=cp.float16)
    for i in range(labels.size):
        y[i, labels[i]] = 1.0
    return y



####################################################################


class MSEloss:
    def __init__(self, ClassSize):
        self.ClassSize = ClassSize
        self.LossVal = 0

    def calcloss(self, Y_pred, Y):
        ttt = Y_pred - OneHot(Y, self.ClassSize)
        tmp = Y_pred.copy()
        iHadm(ttt, ttt, tmp)
        return 0.5 * cp.sum(tmp) / Y.size

    def Backward(self, Y_pred, Y):
        ttt = Y_pred - OneHot(Y, self.ClassSize)
        tmp = Y_pred.copy()
        iHadm(ttt, ttt, tmp)
        self.LossVal = 0.5 * cp.sum(tmp) / Y.size
        return ttt / Y.size

def softmax(Y_pred):
    eX = cp.exp(Y_pred - cp.max(Y_pred,axis=1,keepdims=True))
    return eX / eX.sum(axis=1,keepdims=True)


class CrossEntropyloss:
    def __init__(self):
        self.LossVal = 0
        
    def calcloss(self, Y_pred, Y):
        m = Y_pred.shape[0]
#         p =  softmax(Y_pred)
        eX = cp.exp(Y_pred - cp.max(Y_pred,axis=1,keepdims=True))
        p = eX / eX.sum(axis=1,keepdims=True)
        log_likelihood = -cp.log(p[list(range(m)),Y])
        self.LossVal = cp.sum(log_likelihood) / m
        return self.LossVal
    
    def Backward(self, Y_pred, Y):
        """
        X is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1)
            Note that y is not one-hot encoded vector. 
            It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
        """
        m = Y_pred.shape[0]
#         p =  softmax(Y_pred)
        eX = cp.exp(Y_pred - cp.max(Y_pred,axis=1,keepdims=True))
        p = eX / eX.sum(axis=1,keepdims=True)
        log_likelihood = -cp.log(p[list(range(m)),Y])
        self.LossVal = cp.sum(log_likelihood) / m
#         from calcloss
        grad = p
        grad[list(range(m)),Y] -= 1
        grad = grad/m
        return grad
    

########################################################################


class MaxPool:
    def __init__(self,Xshape, size = 2, stride = 2):
        self.Xshape = Xshape
        self.n, self.d, self.h, self.w = self.Xshape
        n, d, h, w = self.Xshape
        h_out = (h - size) / stride + 1
        w_out = (w - size) / stride + 1
        if not w_out.is_integer() or not h_out.is_integer():
            raise Exception('Invalid output dimension!')
        self.h_out, self.w_out = int(h_out), int(w_out)
        
        self.size = size
        self.stride = stride
        self.Yshape = (self.n, self.d, self.h_out, self.w_out)
    
    def Forward(self,X):
        X_reshaped = X.reshape(self.n * self.d, 1, self.h, self.w)
        self.X_col = im2col_indices(X_reshaped, self.size, self.size, padding=0, stride=self.stride)
        
        self.max_idx = cp.argmax(self.X_col, axis=0)
        
        out = self.X_col[self.max_idx, list(range(self.max_idx.size))]

        out = out.reshape(self.h_out, self.w_out, self.n, self.d)
        self.out = out.transpose(2, 3, 0, 1)

        return self.out
    
    def Backward(self,dout):
        dX_col = cp.zeros_like(self.X_col)
        
        dout_col = dout.transpose(2, 3, 0, 1).ravel()

        dX_col[self.max_idx, list(range(dout_col.size))] = dout_col

        dX = col2im_indices(dX_col, (self.n * self.d, 1, self.h, self.w), self.size, self.size, padding=0, stride=self.stride)
        dX = dX.reshape(self.Xshape)

        return dX    

    def Update(self,lr=0.001):
        return lr
    
        
        
        
class AvgPool:
    def __init__(self,Xshape,size = 2, stride = 2):
        self.Xshape = Xshape
        self.n, self.d, self.h, self.w = self.Xshape
        n, d, h, w = self.Xshape
        h_out = (h - size) / stride + 1
        w_out = (w - size) / stride + 1
        if not w_out.is_integer() or not h_out.is_integer():
            raise Exception('Invalid output dimension!')
        self.h_out, self.w_out = int(h_out), int(w_out)
        
        self.size = size
        self.stride = stride
        self.Yshape = (self.n, self.d, self.h_out, self.w_out)
    
    def Forward(self,X):
        X_reshaped = X.reshape(self.n * self.d, 1, self.h, self.w)
        self.X_col = im2col_indices(X_reshaped, self.size, self.size, padding=0, stride=self.stride)
        
        out = cp.mean(self.X_col, axis=0)

        out = out.reshape(self.h_out, self.w_out, self.n, self.d)
        self.out = out.transpose(2, 3, 0, 1)

        return self.out
    
    def Backward(self,dout):
        dX_col = cp.zeros_like(self.X_col)
        
        dout_col = dout.transpose(2, 3, 0, 1).ravel()

        dX_col[:, list(range(dout_col.size))] = 1. / dX_col.shape[0] * dout_col

        dX = col2im_indices(dX_col, (self.n * self.d, 1, self.h, self.w), self.size, self.size, padding=0, stride=self.stride)
        dX = dX.reshape(self.Xshape)

        return dX    

    def Update(self,lr=0.001):
        return lr
    

    
