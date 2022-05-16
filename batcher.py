from torch import Tensor, LongTensor
from typing import Tuple
import torch

class BatchMeData(Tuple[Tensor,Tensor,Tensor]):
    def __init__(self, tokTuple:tuple, batchSize:int=32):
        self.tokTuple=tokTuple; self.batchSize=batchSize
        self.dataTuple = self.__tensorfy()
        self.batchedTuple=self.__batchify()
        return
    
    def __str__(self):
        return f"Okay, here's the first 10 batches of each set:\nTrain: {self.batchedTuple[0][:10]}\nTest: {self.batchedTuple[1][:10]}\nValidation: {self.batchedTuple[2][:10]}"
        
    def __getitem__(self,idx):
        return self.batchedTuple[idx]
    
    def __tensorfy(self)->Tuple[Tensor,Tensor,Tensor]:
        dt=tuple([])
        for toks in self.tokTuple:
            while len(toks)%self.batchSize!=0: toks.append(0)
            data=[torch.tensor(item) for item in toks]
            dt=(*dt,LongTensor(data))
        return dt
        
    def __batchify(self)->Tuple[Tensor,Tensor,Tensor]:
        bt=tuple([])
        for data in self.dataTuple:
            seqLen = data.size(0)//self.batchSize
            assert len(data)==len(data[:seqLen*self.batchSize])
            bdata=data.view(self.batchSize,seqLen).t().contiguous()
            bt=(*bt,bdata)
        return bt
        
    def __call__(self):
        return self.batchedTuple
