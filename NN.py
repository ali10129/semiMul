# import numpy as np
import cupy as cp
import cupyx

rng = cp.random.default_rng(1)

code = r"""
#define TILE_DIM 16

extern "C"
__inline__ __device__ float semiMul(float* a, float* b) 
{  
    unsigned int d1 = *(unsigned int *)a;
    unsigned int d2 = *(unsigned int *)b;

    unsigned int d3 = 0;

    unsigned char ed1 = (d1) >> 23;
    unsigned char ed2 = (d2) >> 23;

    bool underflow = (ed1 == 0) | (ed2 == 0) | (ed1 + ed2 <= 127);
    bool overflow = ((ed1 + ed2) >= (255 + 127));

    if (underflow)
        d3 = 0;
    else if (overflow)
        d3 = ((d1 ^ d2) | (255 << 23)) & ((256 + 255) << 23);
    else
        d3 = (d1 + d2) - (127 << 23);

    return *(float *)&d3;
}
//////////////////////////////////////////////////////////////////////////////
extern "C" __global__
void semiHadamardScalar(float Scaler, float * A, float * Out, int H, int W)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < H && col < W){
        Out[row * W + col] = semiMul( &Scaler , &A[row * W + col] );
    }
}
////////////////////////////////////////////////////////////////////////////
extern "C" __global__
void semiHadamard(float * A, float * B, float * C, int H, int W)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < H && col < W){
        C[row * W + col] = semiMul( &A[row * W + col] , &B[row * W + col] );
    }
}
//////////////////////////////////////////////////////////////////////////////
extern "C" __global__
void naiveMAC( float* A, float* B, float* C, int m, int k, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float elementC = 0.0f;
    
    for(int i=0; i < k;i++)
    {
            elementC += A[row * k + i] * B[i*n + col];
    }
    C[row*n+col] = elementC;
}

extern "C" __global__
void MAC(float * A, float * B, float * C, int m, int k, int n)
{
        
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int thrX = threadIdx.x;
    int thrY = threadIdx.y;

    //to accumulate partial values of each element in C
    double elementC = 0.0;

    for (int t = 0; t < (k-1)/TILE_DIM +1; t++)
    {
        //Create 2 tiles for matrix A and B at the shared memory
        __shared__ float ATile[TILE_DIM][TILE_DIM];
        __shared__ float BTile[TILE_DIM][TILE_DIM];
        
        //threads to load matrix A to shared memory
        if(row < m && t*TILE_DIM+thrX < k)
            ATile[thrY][thrX] =  A[row*k + t*TILE_DIM+thrX];
        else
            ATile[thrY][thrX] = 0.0;

        //threads to load matrix B to shared memory
        if (t*TILE_DIM+thrY < k && col < n)
            BTile[thrY][thrX] =  B[(t*TILE_DIM+thrY)*n + col];
        else
            BTile[thrY][thrX] = 0.0;
            
        __syncthreads();
        //calculate a partial value of thread element in C
#pragma unroll
        for (int i = 0; i < TILE_DIM; i++){
            //elementC = fma( ATile[thrY][i] , BTile[i][thrX], elementC);
            elementC +=  ATile[thrY][i] * BTile[i][thrX];
        }
        __syncthreads();
    }
    //copy final element value to the C matrix
    if (row < m && col < n)
        C[row*n+col] = (float)elementC;

}
///////////////////////////////////////////////////////////////////////////////

extern "C" __global__
void semiMAC(float * A, float * B, float * C, int m, int k, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int thrX = threadIdx.x;
    int thrY = threadIdx.y;

    //to accumulate partial values of each element in C
    float elementC = 0.0;
    for (int t = 0; t < (k-1)/TILE_DIM +1; t++)
    {
        //Create 2 tiles for matrix A and B at the shared memory
        __shared__ float ATile[TILE_DIM][TILE_DIM];
        __shared__ float BTile[TILE_DIM][TILE_DIM];
        
        //threads to load matrix A to shared memory
        if(row < m && t*TILE_DIM+thrX < k)
            ATile[thrY][thrX] = A[row*k + t*TILE_DIM+thrX];
        else
            ATile[thrY][thrX] = 0.0f;

        //threads to load matrix B to shared memory
        if (t*TILE_DIM+thrY < k && col < n)
            BTile[thrY][thrX] = B[(t*TILE_DIM+thrY)*n + col];
        else
            BTile[thrY][thrX] = 0.0f;

        __syncthreads();

        //calculate a partial value of thread element in C
#pragma unroll
        for (int i = 0; i < TILE_DIM; i++){
            elementC += semiMul( &ATile[thrY][i] , &BTile[i][thrX]);        
        }
        __syncthreads();

    }
    
    //copy final element value to the C matrix
    if (row < m && col < n){
        C[row*n+col] = elementC;
    }
        

}
"""

MAC,semiMAC,semiHadamard,semiHadamardScalar,naiveMAC = None,None,None,None,None
kers = ["MAC", "semiMAC", "semiHadamard", "semiHadamardScalar", "naiveMAC"]
def load_products():
    print("\n********* semiMUL METHOD *******\n")
    raw_mod = cp.RawModule(
        code=code,
        name_expressions=kers,
        options=("--std=c++11",),
    )
    global MAC,semiMAC,semiHadamard,semiHadamardScalar,naiveMAC
    MAC = raw_mod.get_function(kers[0])
    semiMAC = raw_mod.get_function(kers[1])
    semiHadamard = raw_mod.get_function(kers[2])
    semiHadamardScalar = raw_mod.get_function(kers[3])
    naiveMAC = raw_mod.get_function(kers[4])
    
    return MAC,semiMAC,semiHadamard,semiHadamardScalar,naiveMAC

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
    return cols


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
    return x_padded[:, :, padding:-padding, padding:-padding]

###########################################################################################
###########################################################################################

TILE_DIM = 16
def idot(A, W, xOut):
    m = A.shape[0]
    k = A.shape[1]
    n = W.shape[1]
    
    DimGrid = ((n - 1) // TILE_DIM + 1, (m - 1) // TILE_DIM + 1, 1)
    DimBlock = (TILE_DIM, TILE_DIM, 1)
    semiMAC(DimGrid, DimBlock, (A, W, xOut, m, k, n))
    
    
def iHadm(A, B, xOut):
    assert(len(A.shape) == 2)
    H = A.shape[0]
    W = A.shape[1]
    DimGrid = (W, H, 1)
    DimBlock = (1, 1, 1)
    semiHadamard(DimGrid, DimBlock, (A, B, xOut, H,W))    
    
def iHads(scalar, A, xOut):
    assert(len(A.shape) == 2)
    H = A.shape[0]
    W = A.shape[1]
    DimGrid = (W, H, 1)
    DimBlock = (1, 1, 1)
    semiHadamardScalar(DimGrid, DimBlock, (cp.float32(scalar), A, xOut, H,W))  
    return xOut



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
        self.X = cp.zeros((BatchSize, InputSize), dtype=cp.float32)  # null
        self.Y = cp.zeros((BatchSize, OutputSize), dtype=cp.float32)

        self.W = (rng.standard_normal(size=(InputSize, OutputSize), dtype=cp.float32) / 
            cp.sqrt(InputSize / 2)
        ).astype(cp.float32)
        self.B = cp.zeros((1, OutputSize), dtype=cp.float32)

        self.dX = cp.zeros((BatchSize, InputSize), dtype=cp.float32)
        self.dW = cp.zeros((InputSize, OutputSize), dtype=cp.float32)
        self.dB = cp.zeros((1, OutputSize), dtype=cp.float32)

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
        self.W -= iHads(cp.float32(lr), self.dW, self.dW)
        self.B -= iHads(cp.float32(lr), self.dB, self.dB)
        
        

        

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
        ).astype(cp.float32)
        
        self.B = cp.zeros((self.n_filters,1), dtype=cp.float32)
        self.Y = cp.zeros((self.n_x,self.n_filters,self.h_out, self.w_out), dtype=cp.float32)
        
        self.Wcol = cp.array([0], dtype=cp.float32)  # null
        
        self.Xcol = cp.zeros((self.n_filters, self.d_filter, self.h_filter, self.w_filter, self.n_x, self.h_out, self.w_out), dtype=cp.float32)
        
        self.dXcol = cp.zeros((self.d_filter, self.h_filter, self.w_filter, self.n_x, self.h_out, self.w_out), dtype=cp.float32)
        
        self.dX = cp.zeros(self.Xshape, dtype=cp.float32)
        self.dW = cp.zeros(self.Wshape, dtype=cp.float32)
        self.dB = cp.zeros((self.n_filters,1), dtype=cp.float32)
    
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
        iHads(cp.float32(lr), self.dW, self.dW)
        self.W -= self.dW.reshape(self.Wshape)
        self.B -= iHads(cp.float32(lr), self.dB, self.dB)
        # clear dW, dB, dX
        
class Sigmoid:
    def __init__(self, BatchSize: int, OutputSize: int):
        self.Out = cp.array([0], dtype=cp.float32)
        self.dX = cp.zeros((BatchSize, OutputSize), dtype=cp.float32)

    def Forward(self, X):
        self.Out = (-1 * X).copy()
        cp.exp(self.Out, out=self.Out, dtype=cp.float32)
        cp.divide(1.0, (1.0 + self.Out), out=self.Out, dtype=cp.float32)
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
        self.In = cp.array([0], dtype=cp.float32)
        self.Out = cp.array([0], dtype=cp.float32)
        self.dX = cp.array([0], dtype=cp.float32)

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
        self.In = cp.array([0], dtype=cp.float32)
        self.Out = cp.array([0], dtype=cp.float32)
        self.dX = cp.array([0], dtype=cp.float32)
        self.leak = cp.float32(leak)

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
    y = cp.zeros((labels.size, Category), dtype=cp.float32)
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
    

    
