
def main():
    
    import argparse
    import torch
    import wandb
    from torch import nn
    from transformers import AdamW,AutoTokenizer,AutoModelForSequenceClassification,TrainingArguments,Trainer,DataCollatorWithPadding,get_scheduler,get_linear_schedule_with_warmup,TrainerCallback,AutoConfig,EvalPrediction
    from torch.utils.data import DataLoader
    from datasets import load_dataset, Dataset, DatasetDict,ClassLabel
    from transformers import DataCollatorWithPadding
    import numpy as np
    import evaluate
    do_eval=False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from sklearn.model_selection import train_test_split
    import pandas as pd
                    
    
    parser = argparse.ArgumentParser(description='EDOS')
    parser.add_argument('--model_name', type=str, default= 'vinai/bertweet-large', help='name of the deep model')     
    parser.add_argument('--output_dir', type=str, default='vinai/bertweet-large-op', help='path to save model')      
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate')  
    parser.add_argument('--epoch', type=float, default=3, help='number of epochs')     

    args = parser.parse_args()
    checkpoint=args.model_name   #'castorini/afriberta_large'##xlm-roberta-base ##xlm-roberta-large # castorini/afriberta_large #Davlan/naija-twitter-sentiment-afriberta-large#Davlan/xlm-roberta-base- 

    
    
    def preprocess_pandas(data, columns):
        ''' <data> is a dataframe which contain  a <text> column  '''
        df_ = pd.DataFrame(columns=columns)
        df_ = data
                                             # remove special characters
        #df_['text'] = data['text'].str.replace('[^\w\s#@/:%.,_-]', '', flags=re.UNICODE)                           # remove emojis+
        df_['text'] = data['text'].str.replace('[','')
        df_['text'] = data['text'].str.replace(']','')
        df_['text'] = data['text'].str.replace('\n', ' ')
        df_['text'] = data['text'].str.replace('\t', ' ')
        df_['text'] = data['text'].str.replace(' {2,}', ' ', regex=True)                                           # remove 2 or more spaces
        df_['text'] = data['text'].str.lower()
        df_['text'] = data['text'].str.strip()
        df_['text'] = data['text'].replace('\d', '', regex=True)   # remove numbers
    
    
        df_['text'] = data['text'].str.replace("he's", "he is",regex=True)                                         
        df_['text'] = data['text'].str.replace("there's", "there is",regex=True)                                          
        df_['text'] = data['text'].str.replace("We're", "We are",regex=True)                                          
        df_['text'] = data['text'].str.replace("That's", "That is",regex=True)                                          
        df_['text'] = data['text'].str.replace("won't", "will not",regex=True)                                          
        df_['text'] = data['text'].str.replace("they're", "they are",regex=True)                                          
        df_['text'] = data['text'].str.replace("Can't", "Can not",regex=True)                                          
        df_['text'] = data['text'].str.replace("wasn't", "was not",regex=True)                                          
        df_['text'] = data['text'].str.replace("don\x89Ûªt", "do not",regex=True)                                          
        df_['text'] = data['text'].str.replace("aren't", "are not",regex=True)                                          
        df_['text'] = data['text'].str.replace("isn't", "is not",regex=True)                                          
        df_['text'] = data['text'].str.replace("What's", "What is",regex=True)                                          
        df_['text'] = data['text'].str.replace("haven't", "have not",regex=True)                                          
        df_['text'] = data['text'].str.replace("hasn't", "has not",regex=True)                                          
        df_['text'] = data['text'].str.replace("didn't", "did not",regex=True)  
        df_['text'] = data['text'].str.replace("He's", "He is",regex=True)                                          
        df_['text'] = data['text'].str.replace("It's", "It is",regex=True) 
        df_['text'] = data['text'].str.replace("it's", "it is",regex=True)                                          

        df_['text'] = data['text'].str.replace("You're", "You are",regex=True)                                          
        df_['text'] = data['text'].str.replace("I'M", "I am",regex=True)                                          
        df_['text'] = data['text'].str.replace("shouldn't", "should not",regex=True)                                          
        df_['text'] = data['text'].str.replace("wouldn't", "would not",regex=True)                                          
        df_['text'] = data['text'].str.replace("i'm", "I am",regex=True)                                          
        df_['text'] = data['text'].str.replace("I\x89Ûªm", "I am",regex=True)                                          
        df_['text'] = data['text'].str.replace("Here's", "Here is",regex=True)                                          
        df_['text'] = data['text'].str.replace("you\x89Ûªve", "you have",regex=True)                                          
        df_['text'] = data['text'].str.replace("we're", "we are",regex=True)                                          
        df_['text'] = data['text'].str.replace("we're", "we are",regex=True)                                          
        df_['text'] = data['text'].str.replace("what's", "what is",regex=True)                                          
        df_['text'] = data['text'].str.replace("we've", "we have",regex=True)                                          
        df_['text'] = data['text'].str.replace("it\x89Ûªs", "it is",regex=True)                                          
        df_['text'] = data['text'].str.replace("doesn\x89Ûªt", "does not",regex=True)                                          
        df_['text'] = data['text'].str.replace("youve", "you have",regex=True)                                          
        df_['text'] = data['text'].str.replace("who's", "who is",regex=True)                                          
        df_['text'] = data['text'].str.replace("y'all", "you all",regex=True)                                          
        df_['text'] = data['text'].str.replace("would've", "would have",regex=True)                                          
        df_['text'] = data['text'].str.replace("it'll", "it will",regex=True)                                          
        df_['text'] = data['text'].str.replace("we'll", "we will",regex=True)                                          
        df_['text'] = data['text'].str.replace("We've", "We have",regex=True)                                          
        df_['text'] = data['text'].str.replace("he'll", "he will",regex=True)                                          
        df_['text'] = data['text'].str.replace("Weren't", "Were not",regex=True)                                          
        df_['text'] = data['text'].str.replace("Didn't", "Did not",regex=True)                                          
        df_['text'] = data['text'].str.replace("they'll", "they will",regex=True)                                          
        df_['text'] = data['text'].str.replace("they'd", "they would",regex=True)                                          
        df_['text'] = data['text'].str.replace("DON'T", "DO NOT",regex=True)                                          
        df_['text'] = data['text'].str.replace("they've", "they have",regex=True)                                          
        df_['text'] = data['text'].str.replace("they'd", "they would",regex=True)                                          
        df_['text'] = data['text'].str.replace("i'd", "I would",regex=True)                                          
        df_['text'] = data['text'].str.replace("should've", "should have",regex=True)                                          
        df_['text'] = data['text'].str.replace("i'll", "I will",regex=True)                                          
        df_['text'] = data['text'].str.replace("weren't", "were not",regex=True)                                          
        df_['text'] = data['text'].str.replace("They're", "They are",regex=True)                                          
        df_['text'] = data['text'].str.replace("don't", "do not",regex=True)                                          
        df_['text'] = data['text'].str.replace("you're", "you are",regex=True)                                          
        df_['text'] = data['text'].str.replace("i've", "I have",regex=True)                                          
        df_['text'] = data['text'].str.replace("Don't", "do not",regex=True)                                          
        df_['text'] = data['text'].str.replace("I'll", "I will",regex=True)                                          
        df_['text'] = data['text'].str.replace("Ain't", "am not",regex=True)                                          
        df_['text'] = data['text'].str.replace("Don't", "do not",regex=True)                                          
        df_['text'] = data['text'].str.replace("Haven't", "Have not",regex=True)                                          
        df_['text'] = data['text'].str.replace("Could've", "Could have",regex=True)                                          

        df_['text'] = data['text'].str.replace('[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', '', regex=True)                      # remove emails
        df_['text'] = data['text'].str.replace('((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}', '', regex=True)    # remove IP address
        df_['text'] = data['text'].str.replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)          # remove URLs
        df_['text'] = data['text'].str.replace('[#,@,&,<,>,\,/,-]','', regex=True)
   
        df_.drop_duplicates(subset=['text'], keep='first')
        df_ = df_.dropna()
        return df_

    #data_dir= '/home/sanala/Juputer try/afrisent-semeval-2023/modified-training/SubtaskA/'  
    data_dir= '/home/sanala/Juputer try/final_EDOS/TaskC/'
    
    #train = pd.read_csv(data_dir+'train_dev_task_a.csv', header=0)
    #train = pd.read_csv(data_dir+'train_dev_task_c.csv', header=0)
    train = pd.read_csv(data_dir+'train_aug100.csv', header=0)
    #train = pd.read_csv(data_dir+'train_task_c.csv', header=0) 
    #train['text'] = train['text'].apply(lambda x: ' '.join([toxic_misspell_dict.get(word, word) for word in x.split()]))

    train=preprocess_pandas(train, list(train.columns))
    if not do_eval:
        #train = pd.read_csv(data_dir+'augmented_BERTweet_large_xln.csv')
        #train = pd.read_csv(data_dir+'train_task_c.csv')
        train = pd.read_csv(data_dir+'train_aug100.csv', header=0)
        train=preprocess_pandas(train, list(train.columns))
        
        #train = pd.read_csv(data_dir+'oversampled_train_all_tasks.csv')
        #train = pd.read_csv(data_dir+'train_task_a_weak.csv')        


    import pandas as pd
    from imblearn.over_sampling import RandomOverSampler

    # Create a sample DataFrame

    # Define the feature columns and the target column

    X = train['text']
    y = train['label']
    # Define the oversampling strategy
    ros = RandomOverSampler(sampling_strategy='minority')

    # Oversample the minority class
    X_res, y_res = ros.fit_resample(X.values.reshape(-1,1), y)

    # Create a new DataFrame with the oversampled data
    train_sam = pd.DataFrame({'text': X_res[:, 0], 'label': y_res})

    
    
    valid = pd.read_csv(data_dir+'dev_task_c.csv')
    train = train.dropna()
    train = train.dropna()

    valid = valid.dropna()
    label_list = train['label'].unique().tolist()
    num_labels = len(label_list)

    train = Dataset.from_pandas(train)
    valid = Dataset.from_pandas(valid)
    dataset = DatasetDict(
        {
            "train": train,
            "validation": valid
        }
    )
    
    def create_optimizer_and_scheduler(model):

        import transformers
        opt_parameters = []    # To be passed to the optimizer (only parameters of the layers you want to update).
        named_parameters = list(model.named_parameters()) 
        
        # According to AAAMLP book by A. Thakur, we generally do not use any decay 
        # for bias and LayerNorm.weight layers.
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        init_lr = 3.5e-5
        head_lr = 3.6e-5
        lr = init_lr
    
        # === Pooler and regressor ======================================================  
    
        params_0 = [p for n,p in named_parameters if ("pooler" in n or "regressor" in n) 
                and any(nd in n for nd in no_decay)]
        params_1 = [p for n,p in named_parameters if ("pooler" in n or "regressor" in n)
                and not any(nd in n for nd in no_decay)]
    
        head_params = {"params": params_0, "lr": head_lr, "weight_decay": 0.0}    
        opt_parameters.append(head_params)
        
        head_params = {"params": params_1, "lr": head_lr, "weight_decay": 0.01}    
        opt_parameters.append(head_params)
                
        # === 12 Hidden layers ==========================================================
    
        for layer in range(11,-1,-1):        
            params_0 = [p for n,p in named_parameters if f"encoder.layer.{layer}." in n 
                    and any(nd in n for nd in no_decay)]
            params_1 = [p for n,p in named_parameters if f"encoder.layer.{layer}." in n 
                    and not any(nd in n for nd in no_decay)]
        
            layer_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
            opt_parameters.append(layer_params)   
                            
            layer_params = {"params": params_1, "lr": lr, "weight_decay": 0.01}
            opt_parameters.append(layer_params)       
        
            lr *= 0.9  #  0.9     
        
        # === Embeddings layer ==========================================================
    
        params_0 = [p for n,p in named_parameters if "embeddings" in n 
                and any(nd in n for nd in no_decay)]
        params_1 = [p for n,p in named_parameters if "embeddings" in n
                and not any(nd in n for nd in no_decay)]
    
        embed_params = {"params": params_0, "lr": lr, "weight_decay": 0.0} 
        opt_parameters.append(embed_params)
        
        embed_params = {"params": params_1, "lr": lr, "weight_decay": 0.01} 
        opt_parameters.append(embed_params)  
        #num_epochs=3
        #num_training_steps=num_epochs*len(tokenized_datasets["train"])
    
        #scheduler = get_linear_schedule_with_warmup(transformers.AdamW(opt_parameters, lr=init_lr), num_warmup_steps=0,num_training_steps=num_training_steps)

    
        return transformers.AdamW(opt_parameters, lr=init_lr)


    config = AutoConfig.from_pretrained(checkpoint,num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint,config=config).to(device)

    label_to_id = {v: i for i, v in enumerate(label_list)}
    print(label_to_id)
    if label_to_id is not None:
        
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    def preprocess_function(examples):
        # Tokenize the texts
        #print
        texts =(examples['text'],)
        result =  tokenizer(*texts, padding='max_length', max_length=60, truncation=True)
        #print(examples['text'])
        #result = tokenizer(examples['text'], examples['text'], padding=padding, max_length=data_args.max_seq_length, truncation=True)
        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
             result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result



    metric=evaluate.load("f1")
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return metric.compute(predictions=preds, references=p.label_ids,average='macro')

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            # forward pass
            outputs = model(**inputs)
            logits = outputs.get('logits')
            # compute custom loss
            loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([0.242,0.757]).to(device))#0.242,0.757
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    class CustomCallback(TrainerCallback):
    
        def __init__(self, trainer) -> None:
            super().__init__()
            self._trainer = trainer
    
        def on_epoch_end(self, args, state, control, **kwargs):
            if control.should_evaluate:
                control_copy = deepcopy(control)
                self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
                return control_copy




    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    results = []
    logging_steps=100
 
    #for i in range (len(data['train'])):
    for i in range (1):
        train_data=dataset['train']
        tokenized_datasets_train = train_data.map(preprocess_function,batched=True)
     
    
    
        valid_data=dataset['validation']
        tokenized_datasets_valid = valid_data.map(preprocess_function,batched=True)
 
        batch_size=8
        #logging_steps = len(tokenized_datasets["train"]) // batch_size
        logging_steps=100
        training_arg=TrainingArguments('trainerfile',num_train_epochs=args.epoch,logging_steps=logging_steps,learning_rate=args.learning_rate,
                                   per_device_train_batch_size=batch_size)#,per_device_eval_batch_size=batch_size,evaluation_strategy="epoch") #)#,warmup_steps=50)##warmup_steps=50,)#lr_scheduler_type="cosine")# directory where the trained model willbe saved #or we can use (push_to_hub=True) in the TrainingArguments if we want to automatically upload your model to the Hub
        trainer=Trainer(model,args=training_arg,train_dataset=tokenized_datasets_train,
                        eval_dataset=None, #tokenized_datasets_valid, #None ,#if do_eval else None,
                        data_collator=data_collator,tokenizer=tokenizer,
                        compute_metrics=compute_metrics)
        
        
        #optim_scheduler = create_optimizer_and_scheduler(trainer.model) 
        ## override the default optimizer
        #trainer.optimizer = optim_scheduler
        result=trainer.train()

    
    
        #trainer=Trainer(model,args=training_arg,train_dataset=tokenized_datasets_train,eval_dataset=tokenized_datasets_valid,data_collator=data_collator,tokenizer=tokenizer,compute_metrics=compute_metrics)
        #optim_scheduler = create_optimizer_and_scheduler(trainer.model) 
        #trainer.optimizer = optim_scheduler
        #trainer.add_callback(CustomCallback(trainer)) 
        results.append(result)

    trainer.save_model(args.output_dir)
    metrics = result.metrics


if __name__ == "__main__":
    main()
    
    '''
CUDA_VISIBLE_DEVICES=3 python run_train.py \
!python run_train.py  --model_name  'castorini/afriberta_large'  --learning_rate 5e-5 --epoch 3.0 --output_dir 'castorini/afriberta_large-ha'
'''
    
