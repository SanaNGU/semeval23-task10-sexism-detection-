import os
def main():
    
    global SUBMISSION_PATH

    import argparse
    import torch
    from scipy.special import softmax as sx

 
    from torch import nn
    from transformers import AdamW,AutoTokenizer,AutoModelForSequenceClassification,TrainingArguments,Trainer,DataCollatorWithPadding,get_scheduler,get_linear_schedule_with_warmup,TrainerCallback,AutoConfig,EvalPrediction
    import numpy as np
    import evaluate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from sklearn.model_selection import train_test_split
    import pandas as pd
    from sklearn.metrics import classification_report,confusion_matrix

    parser = argparse.ArgumentParser(description='EDOS')
    parser.add_argument('--model_name', type=str, default= 'vinai/bertweet-large', help='name of the deep model')     
    parser.add_argument('--output_dir', type=str, default='vinai/bertweet-large-op', help='path of saved-trained model')     
    parser.add_argument('--results_dir', type=str, default='BERT-with-augmentation', help='path of results')     


    args = parser.parse_args()
    
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.output_dir)
    trainer = Trainer(model)

    class SimpleDataset:
      def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts
  
      def __len__(self):
        return len(self.tokenized_texts["input_ids"])
  
      def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tokenized_texts.items()}
    
    
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

    model_name=args.model_name
    file_name = '/home/sanala/Juputer try/final_EDOS/TaskC/'+'dev_task_c.csv'
    text_column = 'text'

    df_pred = pd.read_csv(file_name)
    #df_pred['text'] = df_pred['text'].apply(lambda x: ' '.join([toxic_misspell_dict.get(word, word) for word in x.split()]))

    df_pred=preprocess_pandas(df_pred, list(df_pred.columns))

    
    ids = df_pred.iloc[:,0].astype('str').tolist()
    pred_texts = df_pred[text_column].astype('str').tolist()

    # Tokenize texts and create prediction data set
    tokenized_texts = tokenizer(pred_texts, truncation=True, padding=True)
    pred_dataset = SimpleDataset(tokenized_texts)

    # Run predictions
    predictions = trainer.predict(pred_dataset)

    # Transform predictions to labels
    preds = predictions.predictions.argmax(-1)
    labels = pd.Series(preds).map(model.config.id2label)
    smax = np.array([list(sx(item)) for item in predictions[0]])

    
   
    target_names = ['2.3 dehumanising attacks & overt sexual objectification', '2.1 descriptive attacks','1.2 incitement and encouragement of harm', '3.1 casual use of gendered slurs, profanities, and insults','4.2 supporting systemic discrimination against women as a group','2.2 aggressive and emotive attacks','3.2 immutable gender differences and gender stereotypes','3.4 condescending explanations or unwelcome advice','3.3 backhanded gendered compliments','4.1 supporting mistreatment of individual women','1.1 threats of harm']
    y_true=df_pred['label']
    
    print(classification_report(y_true, labels, target_names=target_names))
    
    
    from datetime import datetime
    f = open("/home/sanala/Juputer try/final_EDOS/TaskC/output/"+args.results_dir, 'a')
    f.write(f"\n {datetime.today().strftime('%Y-%m-%d %H:%M:%S')} \n")
    f.write("```\n")
    f.write(classification_report(y_true, labels, target_names=target_names))
    f.write("```\n")
    #tn, fp, fn, tp =confusion_matrix(y_true, labels).ravel()
    #f.write(f'tn {tn}, fp {fp}, fn {fn}, tp {tp}')
    #f.write("```\n")
    f.close()
    
    # Create submissions files directory if not available
    if os.path.isdir(args.output_dir):
      print('Data directory found.')
      SUBMISSION_PATH = os.path.join(args.output_dir, 'submission')
      if not os.path.isdir(SUBMISSION_PATH):
        print('Creating submission files directory.')
        os.mkdir(SUBMISSION_PATH)
    else:
      print(args.output_dir + ' is not a valid directory or does not exist!')

    # Create DataFrame with texts, predictions, and labels
    df=pd.DataFrame(list(zip(ids,labels,preds,smax,pred_texts)), columns=['rewire_id', 'label_pred','label_hard','label_soft','text'])
    for i in range (len(df['label_soft'])):
        df['label_soft'][i]=df['label_soft'][i].tolist()
    df.to_csv(os.path.join(SUBMISSION_PATH,'predictions_soft.csv'), index=False)

    df = pd.DataFrame(list(zip(ids,labels)), columns=['rewire_id', 'label_pred'])
    df.to_csv(os.path.join(SUBMISSION_PATH, 'pred_.csv'), index=False)

if __name__ == "__main__":
    main()
    
    '''
CUDA_VISIBLE_DEVICES=3 python run_predict.py --model_name 'castorini/afriberta_large' --language 'ha' --output_dir 'castorini/afriberta_large-ha' 
'''
