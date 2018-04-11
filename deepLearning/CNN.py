import numpy as np
from NN import NN

class CNN(NN):
    def __init__(self,data,hyperParameters,functions):
        pass

    @staticmethod
    def convolutionForward(Aprevious, W, b, hyperParameters):
        pad = hyperParameters['pad']
        assert(len(Aprevious.shape) == 4)
        Apad = np.pad(Aprevious, ((0, 0),(pad, pad),(pad, pad),(0, 0)), 'constant', constant_values=0)
        A = CNN.convolutionForwardWithoutPad(Apad, W, b,hyperParameters['stride'])
        return A
    
    @staticmethod
    def convolutionForwardWithoutPad(Aprevious, W, b, stride = 1):
        assert(len(Aprevious.shape) == 4) # (m, nHpre, nWpre, nCpre)
        assert(len(W.shape) == 4) # (f, f, nCpre, nC)
        assert(Aprevious.shape[3] == W.shape[2]),'number of channel'
        m, nHpre, nWpre, nCpre = Aprevious.shape
        f, f, _, nC = W.shape
        assert(b.shape[-1] == nC),'number of channel is not same'
        b = b.reshape(1,nC) # reshape to broadcasting
        assert(type(stride) == int)
        nH = int((nHpre - f) / stride) + 1
        nW = int((nWpre - f) / stride) + 1
        
        A = np.zeros((m, nH, nW, nC))
        for v in range(nH): # vertical
            for h in range(nW): # horizontal
                vS = v * stride # vertical start
                vE = vS + f # vertical end
                hS = h * stride # horizontal start
                hE = hS + f # horizontal end
                ApreviousSlice = Aprevious[:,vS:vE,hS:hE,:] # (m, f, f, nCpre)
                ApreviousSlice = ApreviousSlice.reshape(m, f, f, nCpre, 1) # the last in shape for broadcasting
                # print('A_p shape:',ApreviousSlice.shape)
                # print('W shape:',W.shape)
                # print('b shape:',b.shape)
                # print('A shape:',A[:,vS:vE,hS:hE,:].shape)
                ASlice = np.multiply(ApreviousSlice,W) + b # (m, f, f, nCpre, 1) * (f, f, nCpre, nC) + (1, nC) -> (m, f, f, nCpre, nC)
                ASlice = np.sum(ASlice, axis= (1,2,3)) # (m, nC)
                A[:,v,h,:] += ASlice # (m, nC) += (m, nC)
        return A

    @staticmethod
    def convolutionBackward(Aprevious, dA, W, b,hyperParameters):
        '''
        Aprevious   --  (m, nHpre, nWpre, nCpre)
        dA          --  (m, nH,    nW   , nC   )
        W           --  (f, f,     nCpre, nC   )
        b           --  (1, 1,     1,     nC   )
        '''
        
        assert(Aprevious.shape[0] == dA.shape[0]),'m not same in Aprevisous and dA'
        assert(dA.shape[3] == W.shape[-1]),'number of channel: dA and W is not same'
        m, nHpre, nWpre, nCpre = Aprevious.shape
        m, nH, nW, nC = dA.shape
        f, f, nCpre, nC = W.shape

        pad = hyperParameters['pad']
        stride = hyperParameters['stride']
        # print('f:',f)
        # print('pad:',pad)
        # print('stride:',stride)
        assert( int((nHpre - f + 2 * pad) / stride) + 1 == nH)
        assert( int((nWpre - f + 2 * pad) / stride) + 1 == nW)

        ApreviousPad = np.pad(Aprevious, ((0, 0),(pad, pad),(pad, pad),(0, 0)), 'constant', constant_values=0) # must use it for dW

        dApreviousPad = np.zeros((m, nHpre + 2 * pad, nWpre + 2 * pad, nCpre))
        dW = np.zeros(W.shape) # (f, f, nCpre, nC)
        db = np.zeros((1,1,1,nC))
        for v in range(nH): # vertical
            for h in range(nW): # horizontal
                vS = v * stride # vertical start
                vE = vS + f # vertical end
                hS = h * stride # horizontal start
                hE = hS + f # horizontal end
                
                dAvh = dA[:,v,h,:] # (m, nC)
                dAvh = dAvh.reshape(m, 1, 1, 1, nC) # (m, 1, 1, 1, nC)

                # compute dA[i-1]
                dApreviousPadSlice = np.multiply(dAvh, W) # (m, 1, 1, 1, nC) * (f, f, nCpre, nC) = (m, f, f, nCpre, nC)
                dApreviousPadSlice = np.sum(dApreviousPadSlice, axis = -1) # zip nC, since ASlice = ApreviousPadSlice * W + b
                dApreviousPad[:,vS:vE,hS:hE,:] += dApreviousPadSlice

                # compute dW
                ApreviousPadSlice = ApreviousPad[:,vS:vE,hS:hE,:] # (m, f, f, nCpre)
                ApreviousPadSlice = ApreviousPadSlice.reshape(m, f, f, nCpre, 1) # reshape for broadcasting
                dWTemp = np.multiply(dAvh, ApreviousPadSlice) # (m, 1, 1, 1, nC) * (m, f, f, nCpre, 1) = (m, f, f, nCpre, nC)
                dW += np.sum(dWTemp, axis = 0)

                # compute db
                db += np.sum(dAvh, axis = 0) # zip number of data

        dAprevious = dApreviousPad[ : , pad:-pad , pad:-pad , : ]
        # print('dApreviousPad shape:',dApreviousPad.shape)
        # print(m, nHpre, nWpre, nCpre)
        assert(dAprevious.shape == (m, nHpre, nWpre, nCpre)),'dAprevious.shape = '+str(dAprevious.shape)
        return dAprevious,dW,db

    @staticmethod
    def maxPoolingForward(Aprevious, f , stride):
        # Aprevious shape is (m, nHpre, nWpre, nCpre)
        assert(len(Aprevious.shape) == 4)
        m, nHpre, nWpre, nCpre = Aprevious.shape
        nH = int((nHpre - f) / stride) + 1
        nW = int((nWpre - f) / stride) + 1
        A = np.zeros((m,nH,nW,nCpre))
        for v in range(nH): # vertical
            for h in range(nW): # horizontal
                vS = v * stride # vertical start
                vE = vS + f # vertical end
                hS = h * stride # horizontal start
                hE = hS + f # horizontal end
                ApreviousSlice = Aprevious[:,vS:vE,hS:hE,:] # (m, f, f, nCpre)
                A[:,v,h,:] += np.max(ApreviousSlice,axis = (1,2)) # take max in (f * f)
        return A

    @staticmethod
    def averagePoolingForward(Aprevious, f, stride):
        # Aprevious shape is (m, nHpre, nWpre, nCpre)
        assert(len(Aprevious.shape) == 4)
        m, nHpre, nWpre, nCpre = Aprevious.shape
        nH = int((nHpre - f) / stride) + 1
        nW = int((nWpre - f) / stride) + 1
        A = np.zeros((m,nH,nW,nCpre))
        for v in range(nH): # vertical
            for h in range(nW): # horizontal
                vS = v * stride # vertical start
                vE = vS + f # vertical end
                hS = h * stride # horizontal start
                hE = hS + f # horizontal end
                ApreviousSlice = Aprevious[:,vS:vE,hS:hE,:] # (m, f, f, nCpre)
                A[:,v,h,:] += 1.0 / f / f * np.sum(ApreviousSlice,axis = (1,2)) # take max in (f * f)
        return A

    @staticmethod
    def poolingForward(Aprevious,hyperParameters):
        f = hyperParameters['f']
        stride = hyperParameters['stride']
        mode = hyperParameters['mode']
        if mode == 'max':
            A = CNN.maxPoolingForward(Aprevious, f, stride)
        elif mode == 'average':
            A = CNN.averagePoolingForward(Aprevious, f, stride)
        return A

    @staticmethod
    def maxPoolingBackward(Aprevious, dA, f, stride):
        '''
        Aprevious   --  (m, nHpre, nWpre, nCpre)
        dA          --  (m, nH,    nW   , nC   )
        '''
        assert(Aprevious.shape[0] == dA.shape[0]),'m not same in Aprevisous and dA'
        m, nHpre, nWpre, nCpre = Aprevious.shape
        m, nH,    nW   , nC    = dA.shape
        assert( int((nHpre - f ) / stride) + 1 == nH)
        assert( int((nWpre - f ) / stride) + 1 == nW)
        
        assert(nCpre == nC),'A is not max Pooling from Aprevious'

        dAprevious = np.zeros(Aprevious.shape) # (m, nHpre, nWpre, nCpre)
        for v in range(nH): # vertical
            for h in range(nW): # horizontal
                vS = v * stride # vertical start
                vE = vS + f # vertical end
                hS = h * stride # horizontal start
                hE = hS + f # horizontal end
                
                dAvh = dA[:,v,h,:] # (m, nC)
                dAvh = dAvh.reshape(m, 1, 1, nC) # (m, 1, 1, nC)

                ApreviousSlice = Aprevious[:,vS:vE,hS:hE,:] # (m, f, f, nCpre)
                ApreviousSliceMaxFF = np.max(ApreviousSlice, axis = (1,2), keepdims=True) # (m, 1, 1, nCpre)
                dApreviousSlice = (ApreviousSlice == ApreviousSliceMaxFF) # (m, f, f, nCpre) , float -> bool
                
                dApreviousSlice = np.multiply(dAvh,dApreviousSlice) # (m, 1, 1, nC) * (m, f, f, nCpre) = (m, f, f, nCpre), since nC == nCpre
                dAprevious[:,vS:vE,hS:hE,:] += dApreviousSlice
        return dAprevious

    def poolingBackward(self):
        pass

    def linearForward(self):
        pass
    def linearBackward(self):
        pass

    def forwardPropagation(self):
        pass
    def backwardPropagation(self):
        pass
    
    # Batch
    def oneBatch(self,X = None):
        pass
    def miniBatch(self):
        pass
    def miniBatchRandom(self):
        pass

if __name__ == 'main':
    pass
else:
    print('CNN is loaded')