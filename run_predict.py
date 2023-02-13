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
    toxic_misspell_dict = {
    's*it': 'shit','sh*t': 'shit',
    's**t': 'shit','shi***': 'shitty','shi**y': 'shitty','shi*': 'shit','shi*s': 'shits',
    'dipsh*t': 'dipshit','s%#*': 'shit','Bu**sh*t': 'Bullshit','bullsh*t': 'bullshit','b******t': 'bullshit',
    'b*llsh*t': 'bullshit','batsh*t': 'batshit','sh*tbird': 'shitbird','sh*tty': 'shitty',
    'bullsh*tter': 'bullshitter','sh@#': 'shit','Sh^t': 'shit','sh^t': 'shit','sh@t': 'shit','$het': 'Shit',
    '$h!t': 'shit','sh1t': 'shit','shlt': 'shit','$h*t': 'shit','bat-s**t': 'bat-shit',
    '$hit': 'shit','sh!te': 'shit','sh!t': 'shit','bullsh1t': 'bullshit','b...llsh1t': 'bullshit',
    's***': 'suck','5h1t': 'shit','sh*thole': 'shithole','bats**t': 'batshit','S**t': 'shit',
    'Batsh*t': 'batshit','Bullsh*t': 'Bullshit','SH*T': 'shit','sh**t': 'shit','sh*t-kicked': 'shit kicked',
    's@!t': 'shit','sh@%%t': 'shit','s#@t': 'shit','#%@$': 'shit','#*@$': 'shit','^@%#': 'fuck',
    '!@#$': 'fuck','s@#t': 'shit','sh@tless': 'shitless','8&@!': 'shit','!@#!!': 'shit','$@%t': 'shit',
    'f@$#': 'fuck','F$$@': 'fuck','F@#$': 'fuck','#@!!%': 'fuck','$%@%ing': 'fucking','F@#k': 'fuck',
    'f@#k': 'fuck','#@!&*': 'fuck','f@#&ing': 'fucking','f@#$^*': 'fucked','F$#@ing': 'fucking',
    '#!$@%^': 'fuck','FU!@#': 'fuck','#%$@er': 'fucker','$@%ing': 'fucking','%#@&ing': 'fucking',
    "*&@k's": 'fuckers','!@#$$!': 'fuck','F#@%ing': 'fucking','F#@*': 'fuck','f@!k': 'fuck',
    '*^@$': 'fuck','$/#@': 'fuck','F!@#in': 'fucking','Fuc@ed': 'fucked','fu@&d': 'fucked','F&@&': 'fuck',
    '$#@^': 'fuck','&@$#': 'fuck','$%@!': 'fuck','fu@#$&': 'fucked','*@#k': 'fuck',
    'F@%!s': 'fucks','fuc$&@ng': 'fucking','f#@k': 'fuck','!$@#%': 'fuck','f******': 'fucking',
    ' f***ing': 'fucking','motherf***ing': 'motherfucking','f***': 'fuck','f***ked': 'fucked',
    'f***ed': 'fucked','fu**ing': 'fucking','clusterf*ck': 'clusterfuck','ratf*cking': 'ratfucking',
    'f*ck': 'fuck','f**k': 'fuck',"f**kin'": 'fucking','F**K': 'fuck','F***': 'fuck','F*ck': 'fuck','f**ks': 'fuck',
    'f**cker': 'fucker','F******': 'fucked','f*&$ing': 'fucking','f*k': 'fuck','F*ggot': 'faggot',
    'F*cks': 'fucks','F*CKING': 'fucking','F*** O**': 'fuck off','f*** o**': 'fuck off','f-up': 'fuck up','F-up': 'fuck up',
    'F@#@CK': 'fuck','F---ck': 'fuck','f---ck': 'fuck','f--ck': 'fuck','F--ck': 'fuck','f-ck': 'fuck',
    'F-ck': 'fuck','f-ckin': 'fucking','fu#$ed': 'fucked','f*$(': 'fuck',' f*$K': 'fuck','f__k': 'fuck',
    'f.ck': 'fuck','fck': 'fuck','Fck': 'fuck','F*ing': 'fucking','f*ing': 'fucking','fukin': 'fucking',
    'fuking': 'fucking','f++k': 'fuck','f*%k': 'fuck','.uck': 'fuck','F@ck': 'fuck','fcuking': 'fucking','a55es': 'asses',
    'a**': 'ass','a*#': 'ass','a******': 'asshole','a*****e': 'asshole','@ss': 'ass','@$$': 'ass',
    'A**': 'ass','A**hole': 'asshole','@##': 'ass','@#$': 'ass', 'a-hole': 'asshole',
    '@sshole': 'asshole', '@ssholes': 'asshole', 'A@@': 'ass',
    'a!@$#$ed': 'assed','ass@s': 'asses','a@#': 'ass','AS^*$@$': 'asses','A#@#$': 'asses','@&&': 'ass',
    'b!tch': 'bitch','b1tch': 'bitch','b*tch': 'bitch',
    'b***h': 'bitch','b***s': 'bitchs','b*th': 'bitch','bit*#^s': 'bitch','b*tt': 'butt','B****': 'bitch','Bit@#$': 'bitch','B***h': 'bitch',
    'Bit*h': 'bitch','bit*h': 'bitch','b****': 'bitch','Bi^@h': 'bitch',
    'B@##S': 'bitchs','Bat-h': 'bitch','b@##$': 'bitch','B@##s': 'bitchs','bit@$': 'bitch','b!t@h': 'bitch',
    'dumb***es': 'dumbasses','Dumb*ss': 'Dumbass','dumba*ss': 'dumbass','broke-a**': 'broke-ass',
    'a***oles': 'assholes','a**holes': 'assholes','da*ned': 'damned','c*#ksukking': 'cock sucking',
    'c***': 'cock','p***y': 'putty','p****': 'putty','P***Y': 'pussy','p***y-grabbing': 'pussy-grabbing',
    'p@$$y': 'pussy','pu$$y': 'pussy','pus$y': 'pussy',
    'pu$sy': 'pussy','p*ssy': 'pussy','pu@#y': 'pussy','p@#$y': 'pussy','puXXy': 'puxxy','puxxy': 'puxxy',
    'N***ga': 'Nigga','s*ck': 'suck','suckees': 'sucker','suckee': 'sucker','s@#k': 'suck',
    's%@': 'suck','s@#K': 'suck','d#$k': 'dick','d@#K': 'dick','d@mn': 'damn','D@mn': 'damn',
    'D@MN': 'damn','da@m': 'damn','p0rn': 'porn','$ex': 'sex','b@stard': 'bastard','b@st@rd': 'bastard','b@#$%^&s': 'bastards',
    'bast@#ds': 'bastards','bas#$@d': 'bastard','b@ssturds': 'bastards','stu*pid': 'stupid','F@KE': 'fake',
    'F@ke': 'fake','N#$@#er': 'nutshell','1%ers': 'very rich people','f@rt': 'fart','d00d': 'dude',
    'n00b': 'noob','ret@rd$': 'retards','ret@rd': 'retard',
    'id-iot': 'idiot','chickens**ts': 'chickenshat','chickens**t': 'chickenshat','0bama': 'obama',
    'ofass': 'of ass','b@t': 'bat','cr@p': 'crap','kr@p': 'crap',
    'c&@p': 'crap','kr@ppy': 'crappy','wh@re': 'whore','b@ll': 'ball',
    'b@ll$': 'balls','6@!!': 'ball','r@pe': 'rape','f@ggot': 'faggot','#@$%': 'cock','su@k': 'suck','r@cist': 'racist',
    'r@ce': 'race','h@ll': 'hell','Isl@m': 'islam','$@rew': 'screwed','scr@wed': 'screwed','j@rk': 'jark',
    's@x': 'sex','idi@t': 'idiot','r@ping': 'raping',
    'V@gina': 'virgina','P^##@*': 'pissed','$k@nk': 'skank','N@zi': 'nazi','MANIA': 'Make America a Nitwit Idiocracy Again',
    'B@t$h!t': 'batshit','bats@3t': 'batshit', 'f@g': 'fag','R@pe': 'rape','s*#@t': 'slot','p@ssw0rd': 'password',
    'p@assword': 'password','Sh*t': 'shit','s**T': 'shit','S**T': 'shit','bullSh*t': 'bullshit',
    'BULLSH*T': 'bullshit','B******T': 'bullshit','Bullsh*tter': 'bullshitter','sh1T': 'shit',
    'Sh1t': 'shit','SH1T': 'shit','$Hit': 'shit','$HIT': 'shit','sh!T': 'shit','Sh!t': 'shit',
    'SH!T': 'shit','Bullsh1t': 'bullshit','S***': 'suck','F***ing': 'fucking','F***ked': 'fucked','Fu**ing': 'fucking',
    'F*CK': 'fuck','F**k': 'fuck','F**ks': 'fuck',
    'F**KS': 'fuck','F*k': 'fuck','F-ckin': 'fucking','F__k': 'fuck','F__K': 'fuck','F.ck': 'fuck','fCk': 'fuck','FcK': 'fuck',
    'FCK': 'fuck','Fukin': 'fucking','f++K': 'fuck','F*%k': 'fuck','A*****e': 'asshole','@SS': 'ass',
    'A-hole': 'asshole','A-Hole': 'asshole','A-HOLE': 'asshole','A@#': 'ass','B!tch': 'bitch','B!TCH': 'bitch',
    'B*tch': 'bitch','B***S': 'bitchs','B*tt': 'butt','DUMBA*SS': 'dumbass','A**holes': 'assholes','A**holeS': 'assholes','C***': 'cock',
    'P***y': 'putty','P****': 'putty','P@$$Y': 'pussy',
    'Pu$$y': 'pussy','PU$$Y': 'pussy','PuS$y': 'pussy','P*ssy': 'pussy','Puxxy': 'puxxy','N00b': 'noob',
    '0Bama': 'obama','B@t': 'bat','Cr@p': 'crap','CR@P': 'crap','Kr@p': 'crap',
    'B@ll': 'ball','P@ssw0rd': 'password','bat****': 'batshit','Bat****': 'batshit','a******s': 'assholes','p****d': 'passed',
    's****': 'shit','S****': 'shit','bull****': 'bullshit','Bull****': 'bullshit','n*****': 'niggar',
    'b*****d': 'bastard','r*****d': 'retarded','f*****g': 'fucking',"a******'s": 'asshole','f****': 'fuck',
    'moth******': 'mother fucker',
    'F******g': 'fucking','n****r': 'niggar','cr*p': 'crap','a-holes': 'asshole','f--k': 'fuck',
    'a**hole': 'asshole','a$$': 'ass','a$s': 'ass','as$': 'ass','@$s': 'ass','@s$': 'ass','$h': 'sh',
    'f***ing': 'fucking','*ss': 'ass','h***': 'hell','p---y': 'pussy',
    "f'n": 'fucking','*&^%': 'shit','a$$hole': 'asshole','p**sy': 'pussy','f---': 'fuck','pi$$': 'piss',
    "f'd up": 'fucked up','c**k': 'cock',
    'a**clown': 'assclown','p___y': 'pussy','sh--': 'shit','f.cking': 'fucking','a--': 'ass','N—–': 'nigga','s*x': 'sex',
    'notalent@$$clown': 'no talent assclown','f--king': 'fucking','a--hole': 'asshole',
    '#whitefragilitycankissmyass': '# white fragility can kiss my ass','N*****': 'niggar','B*****d': 'bastard',
    'F*****G': 'fucking','F****': 'fuck','N****r': 'niggar','Cr*p': 'crap','A-holes': 'asshole','A-Holes': 'asshole',
    'A-HOLES': 'asshole','F--k': 'fuck','F--K': 'fuck','A$$': 'ass','@$S': 'ass','$H': 'sh',
    'F***ing': 'fucking','*SS': 'ass','H***': 'hell','P---y': 'pussy',"F'n": 'fucking',"F'N": 'fucking',
    'A$$hole': 'asshole','A$$HOLE': 'asshole','P**sy': 'pussy','P**SY': 'pussy','F---': 'fuck','Pi$$': 'piss',
    "F'd up": 'fucked up',"F'D UP": 'fucked up','C**k': 'cock','P___y': 'pussy','Sh--': 'shit',
    'SH--': 'shit','A--': 'ass','S*x': 'sex','F--king': 'fucking','A--HOLE': 'asshole',
    'pi**ing': 'pissing',
     '**ok': 'fuck',
     'bi*ch': 'bitch',
     'Sh*ts': 'Shits',
     'Rat****ing': 'fuck',
     '*ds': 'faggots',
     'C*nt': 'Cunt',
     '***ed': 'assholed',
     'h*ll': 'asshole',
     'Re*****s': 'Retards',
    'c*unt': 'cunt',
     'f*rt': 'fuck',
     'p***ing': 'pissing',
     'Pi**ing': 'Pissing',
     'd**m': 'Damn',
     'f***': 'fuck',
     's*': 'Suck',
     'c*nt': 'cunt',
     'dam*d': 'damned',
     'nigg*r': 'nigger',
     'an*l': 'anal',
     'f**t': 'faggot',
     's***': 'shit',
     'H*ll': 'asshole',
     'p***ed': 'pissed',
     'a**ed': 'assholed',
     'd****d': 'fuck',
     'they*you': 'fuck',
     '*****RG': 'fuck',
     'a*s': 'ass',
     'h**l': 'asshole',
     'a*sholes': 'assholes',
     'b****': 'bitch',
     'd*ck': 'fuck',
     'H**L': 'Asshole',
     'mother*cking': 'mother fucking',
     'b*tch': 'bitch',
     'as**in': 'ass',
     'motherfu**ers': 'mother fuckers',
     'bull**it': 'bullshit',
     '****may': '**** may',
     '*Let': '* Let',}
    
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
    #file_name = '/home/sanala/Juputer try/final_EDOS/Data/'+'test_task_a_entries.csv'
    file_name = '/home/sanala/Juputer try/final_EDOS/'+'dev_task_a.csv'

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
    
    target_names = ['not sexsit','sexist']
 
    y_true=df_pred['label']
    
    
    print(classification_report(y_true, labels, target_names=target_names))
    
    
    from datetime import datetime
    f = open("/home/sanala/Juputer try/final_EDOS/output/"+args.results_dir, 'a')
    f.write(f"\n {datetime.today().strftime('%Y-%m-%d %H:%M:%S')} \n")
    f.write("```\n")
    f.write(classification_report(y_true, labels, target_names=target_names))
    f.write("```\n")
    tn, fp, fn, tp =confusion_matrix(y_true, labels).ravel()
    f.write(f'tn {tn}, fp {fp}, fn {fn}, tp {tp}')
    f.write("```\n")
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
