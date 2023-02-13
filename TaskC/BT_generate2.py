import fairseq
import nlpaug
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
back_translation_aug_1 = naw.BackTranslationAug(
    from_model_name='facebook/wmt19-en-de',
    to_model_name='facebook/wmt19-de-en',
    device='cuda', verbose=1)
back_translation_aug_2 = naw.BackTranslationAug(
    from_model_name='facebook/wmt19-en-ru',
    to_model_name='facebook/wmt19-ru-en',
    device='cuda', verbose=1)
import pandas as pd
from sklearn.model_selection import train_test_split
data_dir= '/home/sanala/Juputer try/final_EDOS/TaskC/'
train = pd.read_csv(data_dir+'train_task_c.csv', header=0)
aug_text_6=[]
 

for i in range(len(train)):
    
    if train['label'][i] == '3.1 casual use of gendered slurs, profanities, and insults':
        
        bt1=back_translation_aug_1.augment(train['text'][i])
        bt2=back_translation_aug_2.augment(bt1)
        aug_text_6.append(bt1)
        aug_text_6.append(bt2)
   



BT_6_df = pd.DataFrame()

BT_6_df['text']= aug_text_6



BT_6_df['label']='3.1 casual use of gendered slurs, profanities, and insults'




BT_6_df.to_csv('/home/sanala/Juputer try/final_EDOS/TaskC/aug.csv', index=False)

