import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
import numpy as np
DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
def choose_from_top_k_top_n(probs, k=50, p=0.8):##we can also make k and p hyperparameter
	ind = np.argpartition(probs, -k)[-k:]
	top_prob = probs[ind]
	top_prob = {i: top_prob[idx] for idx,i in enumerate(ind)}
	sorted_top_prob = {k: v for k, v in sorted(top_prob.items(), key=lambda item: item[1], reverse=True)}
	
	t=0
	f=[]
	pr = []
	for k,v in sorted_top_prob.items():
		t+=v
		f.append(k)
		pr.append(v)
		if t>=p:
			break
	top_prob = pr / np.sum(pr)
	token_id = np.random.choice(f, 1, p = top_prob)

	return int(token_id)

def generate(tokenizer, model, sentences, label,device):
	gen_text=[]
	with torch.no_grad():
	  for idx in range(sentences):
		  finished = False
		  cur_ids = torch.tensor(tokenizer.encode(label)).unsqueeze(0).to('cpu')
		  for i in range(100):
			  outputs = model(cur_ids, labels=cur_ids)
			  loss, logits = outputs[:2]

			  softmax_logits = torch.softmax(logits[0,-1], dim=0)

			  #if i < 5:
				  #n = 10
			  #else:
				 # n = 5

			  next_token_id = choose_from_top_k_top_n(softmax_logits.to('cpu').numpy()) #top-k-top-n sampling
			  cur_ids = torch.cat([cur_ids, torch.ones((1,1)).long().to('cpu') * next_token_id], dim = 1)

			  if next_token_id in tokenizer.encode('<|endoftext|>'):
				  finished = True
				  break

		  if finished:       
			  output_list = list(cur_ids.squeeze().to('cpu').numpy())
			  output_text = tokenizer.decode(output_list[ : -1])#but the last
			  print (output_text)
			  gen_text.append(output_text)
              
		  else:
			  output_list = list(cur_ids.squeeze().to('cpu').numpy())
			  output_text = tokenizer.decode(output_list[ : -1])
			  print (output_text)
			  gen_text.append(output_text)
   
	return gen_text
def load_models(model_name):
	"""
	Summary:
		Loading the trained model
	"""
	print ('Loading Trained GPT-2 Model')
	tokenizer = AutoTokenizer.from_pretrained('gpt2-xl') # gpt2-xl #gpt2-medium  #gpt2-large
	model = AutoModelForCausalLM.from_pretrained('gpt2-xl')# gpt2-xl #gpt2-medium  #gpt2-large
	model_path = model_name
	model.load_state_dict(torch.load(model_path))
	return tokenizer, model

model_name='mygpt2xlmodel.pt' 
sentences=1500
 
SENTENCES =sentences
MODEL_NAME =model_name
LABEL = 'not sexist'

TOKENIZER, MODEL = load_models(MODEL_NAME)
print(DEVICE)
gen_text=generate(TOKENIZER, MODEL, SENTENCES, LABEL,DEVICE)
import pandas as pd
df = pd.DataFrame (gen_text)
df.to_csv('/home/sanala/Juputer try/final_EDOS/augmented_data_not.csv', index=False)