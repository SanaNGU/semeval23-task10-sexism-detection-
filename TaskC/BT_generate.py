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
aug_text_1=[]
aug_text_2=[]
aug_text_3=[]
aug_text_4=[]
aug_text_5=[]
aug_text_6=[]
aug_text_7=[]
aug_text_8=[]
aug_text_9=[]
aug_text_10=[]
aug_text_11=[]

for i in range(len(train)):
    if train['label'][i] == '1.1 threats of harm':
        bt1=back_translation_aug_1.augment(train['text'][i])
        bt2=back_translation_aug_2.augment(bt1)
        aug_text_1.append(bt1)
        aug_text_1.append(bt2)
    elif train['label'][i] == '1.2 incitement and encouragement of harm':
        bt1=back_translation_aug_1.augment(train['text'][i])
        bt2=back_translation_aug_2.augment(bt1)
        aug_text_2.append(bt1)
        aug_text_2.append(bt2)
    elif train['label'][i] == '2.1 descriptive attacks':
        bt1=back_translation_aug_1.augment(train['text'][i])
        bt2=back_translation_aug_2.augment(bt1)
        aug_text_3.append(bt1)
        aug_text_3.append(bt2)
    elif train['label'][i] == '2.2 aggressive and emotive attacks':
        bt1=back_translation_aug_1.augment(train['text'][i])
        bt2=back_translation_aug_2.augment(bt1)
        aug_text_4.append(bt1)
        aug_text_4.append(bt2)
    elif train['label'][i] == '2.3 dehumanising attacks & overt sexual objectification':
        bt1=back_translation_aug_1.augment(train['text'][i])
        bt2=back_translation_aug_2.augment(bt1)
        aug_text_5.append(bt1)
        aug_text_5.append(bt2)
    elif train['label'][i] == '3.1 casual use of gendered slurs, profanities, and insults':
        
        bt1=back_translation_aug_1.augment(train['text'][i])
        bt2=back_translation_aug_2.augment(bt1)
        aug_text_6.append(bt1)
        aug_text_6.append(bt2)
    elif train['label'][i] == '3.2 immutable gender differences and gender stereotypes':
        bt1=back_translation_aug_1.augment(train['text'][i])
        bt2=back_translation_aug_2.augment(bt1)
        aug_text_7.append(bt1)
        aug_text_7.append(bt2)
    elif train['label'][i] == '3.3 backhanded gendered compliments':
        bt1=back_translation_aug_1.augment(train['text'][i])
        bt2=back_translation_aug_2.augment(bt1)
        aug_text_8.append(bt1)
        aug_text_8.append(bt2)
    elif train['label'][i] == '3.4 condescending explanations or unwelcome advice':
        bt1=back_translation_aug_1.augment(train['text'][i])
        bt2=back_translation_aug_2.augment(bt1)
        aug_text_9.append(bt1)
        aug_text_9.append(bt2)
    elif train['label'][i] == '4.1 supporting mistreatment of individual women':
        bt1=back_translation_aug_1.augment(train['text'][i])
        bt2=back_translation_aug_2.augment(bt1)
        aug_text_10.append(bt1)
        aug_text_10.append(bt2)
    elif train['label'][i] == '4.2 supporting systemic discrimination against women as a group':
        bt1=back_translation_aug_1.augment(train['text'][i])
        bt2=back_translation_aug_2.augment(bt1)
        aug_text_11.append(bt1)
        aug_text_11.append(bt2)


BT_1_df = pd.DataFrame()
BT_2_df = pd.DataFrame()
BT_3_df = pd.DataFrame()
BT_4_df = pd.DataFrame()
BT_5_df = pd.DataFrame()
BT_6_df = pd.DataFrame()
BT_7_df = pd.DataFrame()
BT_8_df = pd.DataFrame()
BT_9_df = pd.DataFrame()
BT_10_df = pd.DataFrame()
BT_11_df = pd.DataFrame()

BT_1_df['text']=aug_text_1
BT_2_df ['text']= aug_text_2
BT_3_df ['text']= aug_text_3
BT_4_df ['text']= aug_text_4
BT_5_df['text']=aug_text_5
BT_6_df ['text']= aug_text_6
BT_7_df ['text']= aug_text_7
BT_8_df ['text']= aug_text_8
BT_9_df ['text']= aug_text_9
BT_10_df ['text']= aug_text_10
BT_11_df ['text']= aug_text_11

BT_1_df['label']='1.1 threats of harm'
BT_2_df['label']='1.2 incitement and encouragement of harm'
BT_3_df['label']='2.1 descriptive attacks'
BT_4_df['label']='2.2 aggressive and emotive attacks'
BT_5_df['label']='2.3 dehumanising attacks & overt sexual objectification'
BT_6_df['label']='3.1 casual use of gendered slurs, profanities, and insult'
BT_7_df['label']='3.2 immutable gender differences and gender stereotypes'
BT_8_df['label']='3.3 backhanded gendered compliments'
BT_9_df['label']='3.4 condescending explanations or unwelcome advice'
BT_10_df['label']='4.1 supporting mistreatment of individual women'
BT_11_df['label']='4.2 supporting systemic discrimination against women as a group'

BT_taskb= pd.concat([BT_1_df,BT_2_df,BT_3_df,BT_4_df,BT_5_df,BT_6_df,BT_7_df,BT_8_df,BT_9_df,BT_10_df,BT_11_df])

BT_taskb.to_csv('/home/sanala/Juputer try/final_EDOS/TaskC/augmented_data_bt.csv', index=False)

