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

    parser = argparse.ArgumentParser(description='EDOS')
    parser.add_argument('--model_name', type=str, default= 'vinai/bertweet-large', help='name of the deep model')     
    parser.add_argument('--output_dir', type=str, default='vinai/bertweet-large-op', help='path of saved-trained model')     
 

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
    


    model_name=args.model_name
    #file_name = '/home/sanala/Juputer try/final_EDOS/Data/'+'test_task_a_entries.csv'
    file_name = '/home/sanala/Juputer try/final_EDOS/'+'augmented_data.csv'

    
    text_column = 'text'

    df_pred = pd.read_csv(file_name)
 
 
    
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
    df=pd.DataFrame(list(zip(labels,preds,smax,pred_texts)), columns=['label_pred','label_hard','label_soft','text'])
    for i in range (len(df['label_soft'])):
        df['label_soft'][i]=df['label_soft'][i].tolist()
    df.to_csv(os.path.join(SUBMISSION_PATH,'predictions_soft.csv'), index=False)

    df = pd.DataFrame(list(zip(pred_texts,labels)), columns=['text','label_pred'])
    df.to_csv(os.path.join(SUBMISSION_PATH, 'pred_.csv'), index=False)

if __name__ == "__main__":
    main()
    
    '''
CUDA_VISIBLE_DEVICES=3 python run_predict.py --model_name 'castorini/afriberta_large' --language 'ha' --output_dir 'castorini/afriberta_large-ha' 
'''
