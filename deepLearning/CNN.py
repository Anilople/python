import numpy as np
from NN import NN

class CNN(NN):
    def __init__(self,data,hyperParameters,functions):
        pass

    @staticmethod
    def convolutionForward(Aprevious, W, b, hyperParameters):
        pad = hyperParameters['pad']
        assert(len(Aprevious.shape) == 4)
        APad = np.pad(Aprevious, ((0, 0),(pad, pad),(pad, pad),(0, 0)), 'constant', constant_values=0)
        A = CNN.convolutionForwardWithoutPad(APad, W, b,hyperParameters['stride'])
        return A
    
    @staticmethod
    def convolutionForwardWithoutPad(Aprevious, W, b, stride = 1):
        assert(len(Aprevious.shape) == 4) # (m, nHPre, nWPre, nCPre)
        assert(len(W.shape) == 4) # (f, f, nCpre, nC)
        assert(Aprevious.shape[3] == W.shape[2]),'number of channel'
        m, nHPre, nWPre, nCPre = Aprevious.shape
        f, f, _, nC = W.shape
        assert(b.shape[-1] == nC),'number of channel is not same'
        b = b.reshape(1,nC) # reshape to broadcasting
        assert(type(stride) == int)
        nH = int((nHPre - f) / stride) + 1
        nW = int((nWPre - f) / stride) + 1
        
        A = np.zeros((m, nH, nW, nC))
        for v in range(nH): # vertical
            for h in range(nW): # horizontal
                vS = v * stride # vertical start
                vE = vS + f # vertical end
                hS = h * stride # horizontal start
                hE = hS + f # horizontal end
                ApreviousSlice = Aprevious[:,vS:vE,hS:hE,:] # (m, f, f, nCPre)
                ApreviousSlice = ApreviousSlice.reshape(m, f, f, nCPre, 1) # the last in shape for broadcasting
                # print('A_p shape:',ApreviousSlice.shape)
                # print('W shape:',W.shape)
                # print('b shape:',b.shape)
                # print('A shape:',A[:,vS:vE,hS:hE,:].shape)
                ASlice = np.multiply(ApreviousSlice,W) + b# (m, f, f, nCPre, 1) * (f, f, nCpre, nC) + (1, nC) -> (m, f, f, nCpre, nC)
                ASlice = np.sum(ASlice, axis= (1,2,3)) # (m, nC)
                A[:,v,h,:] += ASlice # (m, nC) += (m, nC)
        return A

    def convolutionBackward(self):
        pass

    @staticmethod
    def maxPooling(Aprevious, f , stride):
        # Aprevious shape is (m, nHPre, nWPre, nCPre)
        assert(len(Aprevious.shape) == 4)
        m, nHPre, nWPre, nCPre = Aprevious.shape
        nH = int((nHPre - f) / stride) + 1
        nW = int((nWPre - f) / stride) + 1
        A = np.zeros((m,nH,nW,nCPre))
        for v in range(nH): # vertical
            for h in range(nW): # horizontal
                vS = v * stride # vertical start
                vE = vS + f # vertical end
                hS = h * stride # horizontal start
                hE = hS + f # horizontal end
                ApreviousSlice = Aprevious[:,vS:vE,hS:hE,:] # (m, f, f, nCPre)
                A[:,v,h,:] += np.max(ApreviousSlice,axis = (1,2)) # take max in (f * f)
        return A

    @staticmethod
    def averagePooling(Aprevious, f , stride):
        # Aprevious shape is (m, nHPre, nWPre, nCPre)
        assert(len(Aprevious.shape) == 4)
        m, nHPre, nWPre, nCPre = Aprevious.shape
        nH = int((nHPre - f) / stride) + 1
        nW = int((nWPre - f) / stride) + 1
        A = np.zeros((m,nH,nW,nCPre))
        for v in range(nH): # vertical
            for h in range(nW): # horizontal
                vS = v * stride # vertical start
                vE = vS + f # vertical end
                hS = h * stride # horizontal start
                hE = hS + f # horizontal end
                ApreviousSlice = Aprevious[:,vS:vE,hS:hE,:] # (m, f, f, nCPre)
                A[:,v,h,:] += 1.0 / f / f * np.sum(ApreviousSlice,axis = (1,2)) # take max in (f * f)
        return A

    @staticmethod
    def poolingForward(Aprevious,):
        pass
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