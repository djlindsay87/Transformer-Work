import warnings
warnings.filterwarnings("ignore")
from pytorch_model_summary import summary
import pickle
from os.path import exists
import sys

import torch

import the_beast
import batcher
import n_net
import trevor

#URIs
URIs={'theChats' :"spotify:artist:1aQ7P3HtKOQFW16ebjiks1",
'theMarsVolta' :"spotify:artist:75U40yZLLPglFgXbDVnmVs",
'theBeatles':'spotify:artist:3WrFJ7ztbogyGnTHbHJFl2',
'kendrick': 'spotify:artist:2YZyLoL8N0Wb9xBt1NhZWg'}

def main():

    args=sys.argv[1:]

    if (len(args) >=2 and args[0]=='-token' and args[1] in URIs):
        aToken=args[1]
        bToken=URIs[aToken]
    else: bToken=None
    if (len(args)>=4 and args[2]=='-range' and isinstance(args[3],tuple)):
        if isinstance(args[3][0], int) and isinstance(args[3][1], int):
            yRange=args[3]
    else: yRange=None
    if (len(args)==6 and args[4]=='-samples' and isinstance(args[5], int)):
        nSongs=args[5]
    else: nSongs=None
    
    while not bToken:
        aToken=input("Enter an artist token: ")
        if aToken in URIs:
            bToken = URIs[aToken]
            break
                
    if not exists(f'{aToken}-model.pkl'):
        if not yRange:
            veri0=input("Enter a range of years to sample between? [y]/n: ")
            if (veri0.lower()=='y')|(veri0.lower()=='yes'):
                yRange=input("Enter a tuple with two years in it: ")
                while not (isinstance(yRange, tuple)) and (isinstance(yRange[0], int) and isinstance(yRange[1], int)):
                    yRange = input("Try again: ")
            else: yRange=None
        
        try:
            theSet = the_beast.TheBeast(bToken, nSongs=nSongs, yRange=yRange)
        except:
            print('The Beast is sleeping... Check your token or range :(')
            return 1
            
        theBatch = batcher.BatchMeData(theSet(), batchSize=32)
        train_data, test_data, val_data = theBatch()
        
        veri=input("Do you want to look at the data? [y]/n: ")
        if (veri.lower()=='y')|(veri.lower()=='yes'):
            print(theBatch)
    
        ntokens = len(theSet) # size of vocabulary
        emsize = 200  # embedding dimension
        nhead = 4  # number of heads in nn.MultiheadAttention
        d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
        nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        bptt=n_net.bptt # set in n_net b/c 
        dropout = 0.2  # dropout probability
    
        theModel = n_net.TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, bptt, dropout).to(n_net.device)
        print('model created')
    
        veri2=input("Do you wish to see a model summary? [y]/n: ")
        if (veri2.lower()=='y')|(veri2.lower()=='yes'):
            print(summary(theModel, train_data, n_net.generate_square_subsequent_mask(train_data.size(0)), batch_size=theBatch.batchSize, show_input=True, show_hierarchical=True))
        
        trevor.TrainEval(theModel, bptt, 5, ntokens, train_data, test_data, val_data)

        pickle.dump(theSet, open(f'{aToken}-set.pkl','wb'))
        pickle.dump(theModel, open(f'{aToken}-model.pkl', 'wb'))
        print("Model pickled!")
    else: ### ATTEMPT TO GET THIS SHIT GENERATING! ###############################################
        pickledModel = pickle.load(open(f'{aToken}-model.pkl', 'rb'))
        print(pickledModel) # model summary... Is this pickled right?
        growingSet=pickle.load(open(f'{aToken}-set.pkl','rb'))

        def generate(input_sentence:str)->str:
            pickledModel.eval() # Go into evaluation mode
            print(len(growingSet)) # Vocab size (nTokens)
            src=growingSet.encode(input_sentence.lower()) # converts your input to numerical tokens
            SOS=growingSet.textTokDict['<sos>']; EOS=growingSet.textTokDict['<eos>'] # grab the sos and eos tokens
            entry=[SOS]+src; seqLen=len(entry) # add sos token to front of entry, assigns sequence length
            y_input=torch.tensor([entry], device=n_net.device).t() # tensorfies the input, seqlen is along the 0 dim
            with torch.no_grad(): # I feel this is an appropriate time to use no grad but not sure why?
                for i in range(0, 300): # Enough range to hopefully get an eos token
                    # mask is growing with the sequence length... Is this my misunderstanding?
                    src_mask = n_net.generate_square_subsequent_mask(seqLen+i).to(n_net.device)
                    output = pickledModel(y_input,src_mask) # output shape [seqlen+i, 1, nTokens]
                    # So yes, output is the same size of y_input, with the added dimensionality of vocab (nTokens)
                    print(output.shape) # proof
                    # Okay another point of potential misunderstanding, I'll try and be concise:
                    # is this output a tensor of probabilities in the 2 dim? does index corresponds to vocab?
                    nextIndex=output[-1][0][:].argmax() # if so wouldn't the weights of the desired next item
                    # be at the very bottom of the output?
                    print(nextIndex) # cat in 0 dim, but we gotta add dimensionality to nextIndex
                    # this is hacky, I know
                    y_input=torch.cat((y_input, nextIndex.unsqueeze(dim=-1).unsqueeze(dim=-1)), dim=0) 
                    if nextIndex.item() == EOS: #if output is eos token, we break
                        break
            # returns a decoded string of text, starting with your input            
            return " ".join(growingSet.decode(y_input.view(-1).tolist())) 
        
        print(generate("Play us something different")) # an attempt generates repeating text
            


            

if __name__ == '__main__':
    main()

'''
    def evaluate(self, model: nn.Module, eval_data: Tensor) -> float:
        model.eval()  # turn on evaluation mode
        total_loss = 0.
        src_mask = n_net.generate_square_subsequent_mask(self.bptt).to(n_net.device)
        with torch.no_grad():
            for i in range(0, eval_data.size(0) - 1, self.bptt):


                seqLen=min(bptt, len(eval_data)-1-i)
                data= eval_data[i:i+seqLen]
                targets=eval_data[i+1:i+1+seqLen].reshape(-1)


                batch_size = data.size(0)
                if batch_size != self.bptt:
                    src_mask = src_mask[:batch_size, :batch_size]
                output = model(data, src_mask)
                output_flat = output.view(-1, self.ntokens)
                total_loss += batch_size * self.criterion(output_flat, targets).item()
        return total_loss / (len(eval_data) - 1)


'''