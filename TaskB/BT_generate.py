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
data_dir= '/home/sanala/Juputer try/final_EDOS/TaskB/'
train = pd.read_csv(data_dir+'train_task_b.csv', header=0)
aug_text_1=[]
aug_text_2=[]
aug_text_3=[]
aug_text_4=[]

for i in range(len(train)):
    if train['label'][i] == '1. threats, plans to harm and incitement':
        bt1=back_translation_aug_1.augment(train['text'][i])
        bt2=back_translation_aug_2.augment(bt1)
        aug_text_1.append(bt1)
        aug_text_1.append(bt2)
    elif train['label'][i] == '2. derogation':
        bt1=back_translation_aug_1.augment(train['text'][i])
        bt2=back_translation_aug_2.augment(bt1)
        aug_text_2.append(bt1)
        aug_text_2.append(bt2)
    elif train['label'][i] == '3. animosity':
        bt1=back_translation_aug_1.augment(train['text'][i])
        bt2=back_translation_aug_2.augment(bt1)
        aug_text_3.append(bt1)
        aug_text_3.append(bt2)
    elif train['label'][i] == '4. prejudiced discussions':
        bt1=back_translation_aug_1.augment(train['text'][i])
        bt2=back_translation_aug_2.augment(bt1)
        aug_text_4.append(bt1)
        aug_text_4.append(bt2)

BT_1_df = pd.DataFrame()
BT_2_df = pd.DataFrame()
BT_3_df = pd.DataFrame()
BT_4_df = pd.DataFrame()

BT_1_df['text']=aug_text_1
BT_2_df ['text']= aug_text_2
BT_3_df ['text']= aug_text_3
BT_4_df ['text']= aug_text_4


BT_1_df['label']='1. threats, plans to harm and incitement'
BT_2_df['label']='2. derogation'
BT_3_df['label']='3. animosity'
BT_4_df['label']='4. prejudiced discussions'



BT_taskb= pd.concat([BT_1_df,BT_2_df,BT_3_df,BT_4_df])

BT_taskb.to_csv('/home/sanala/Juputer try/final_EDOS/TaskB/augmented_data_bt.csv', index=False)

