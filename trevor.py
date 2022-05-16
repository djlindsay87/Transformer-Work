import math, copy, time
import torch
from torch import nn, Tensor

import n_net

class TrainEval():
	def __init__(self, model, bptt, epochs, ntokens, train_data, test_data, val_data):
		self.model=model
		self.bptt=bptt
		self.ntokens=ntokens
		self.train_data=train_data
		self.test_data=test_data
		self.val_data=val_data
		
		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = torch.optim.SGD(self.model.parameters(), lr=5)
		self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)

		best_model = None
		best_val_loss = float('inf')
		for epoch in range(1, epochs + 1):
			epoch_start_time = time.time()
			self.train(self.model, epoch)
			val_loss = self.evaluate(self.model, self.val_data)
			val_ppl = math.exp(val_loss)
			elapsed = time.time() - epoch_start_time
			print('-' * 89)
			print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
				f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
			print('-' * 89)
			
			if val_loss < best_val_loss:
				best_val_loss = val_loss
				best_model = copy.deepcopy(self.model)

			self.scheduler.step()
	  
		test_loss = self.evaluate(best_model, self.test_data)
		test_ppl = math.exp(test_loss)
		print('=' * 89)
		print(f'| End of training | test loss {test_loss:5.2f} | '
			f'test ppl {test_ppl:8.2f}')
		print('=' * 89)
	
	def train(self, model: nn.Module, epoch:int, log_interval = 1):
		print('training...')
		model.train()  # turn on train mode
		total_loss = 0.
		start_time = time.time()
		src_mask = n_net.generate_square_subsequent_mask(self.bptt).to(n_net.device)
		
		num_batches = len(self.train_data) // self.bptt
		for batch, i in enumerate(range(0, self.train_data.size(0) - 1, self.bptt)):
			data, targets = n_net.splitBatch(self.train_data, i)
			batch_size = data.size(0)
			if batch_size != self.bptt:  # only on last batch
				src_mask = src_mask[:batch_size, :batch_size]
				print("Data:",data.shape, data, "\nMask:", src_mask.shape)
			output = model(data, src_mask)
			if batch_size != self.bptt: print("Output:",output.shape, output,"\nTarget:",targets.shape,targets)
			loss = self.criterion(output.view(-1, self.ntokens), targets)
			
			self.optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
			self.optimizer.step()
			
			total_loss += loss.item()
			if batch % log_interval == 0 and batch > 0:
				lr = self.scheduler.get_last_lr()[0]
				ms_per_batch = int((time.time() - start_time) * 1000 // log_interval)
				cur_loss = total_loss / log_interval
				ppl = math.exp(cur_loss)
				print(f'| epoch {epoch:3d} | '
					f'{batch:5d}/{num_batches:5d} batches | '
					f'lr {lr:02.2f} | ms/batch {ms_per_batch:3d} | '
					f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
				total_loss = 0
				start_time = time.time()
		return
	
	def evaluate(self, model: nn.Module, eval_data: Tensor) -> float:
		model.eval()  # turn on evaluation mode
		total_loss = 0.
		src_mask = n_net.generate_square_subsequent_mask(self.bptt).to(n_net.device)
		with torch.no_grad():
			for i in range(0, eval_data.size(0) - 1, self.bptt):
				data, targets = n_net.splitBatch(eval_data, i)
				batch_size = data.size(0)
				if batch_size != self.bptt:
					src_mask = src_mask[:batch_size, :batch_size]
				output = model(data, src_mask)
				output_flat = output.view(-1, self.ntokens)
				total_loss += batch_size * self.criterion(output_flat, targets).item()
		return total_loss / (len(eval_data) - 1)
	

		
	
'''
def generate(model: nn.Module, src_text:str):
    src=BeatleSet.encode(src_text.lower())
    SOS=BeatleSet.textTokDict['<sos>'] ; EOS=BeatleSet.textTokDict['<eos>']
    print(src)
    model.eval(); entry=[SOS]+src
    y_input=torch.tensor([entry], dtype=torch.long, device=device)
    for i in range(50):
        with torch.no_grad():
            src_mask=generate_square_subsequent_mask(y_input.size(1)).to(device)
            if i%49==0: print(y_input.shape,y_input,'\n',src_mask.shape,src_mask)
            pred=model(y_input.t(), src_mask)
            if i%49==0: print(pred)
            next_item = pred.topk(1)[1].view(-1)[-1].item()
            next_item = torch.tensor([[next_item]], device=device)
            if i%49==0: print(next_item,next_item.shape)
            y_input=torch.cat((y_input, next_item), dim=1)
            
            if next_item.view(-1).item() == EOS:
                break
    return " ".join(BeatleSet.decode(y_input.view(-1).tolist()))
'''