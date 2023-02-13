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
data_dir= '/home/sanala/Juputer try/final_EDOS/'
train = pd.read_csv(data_dir+'train_task_a.csv', header=0)
aug_text_sexist=[]
aug_text_not_sexist=[]
for i in range(len(train)):
    if train['label'][i] == 'sexist':
        bt1=back_translation_aug_1.augment(train['text'][i])
        bt2=back_translation_aug_2.augment(bt1)
        print(bt2)
        aug_text_sexist.append(bt1)
        aug_text_sexist.append(bt2)
    elif train['label'][i] == 'not sexist':
        bt1=back_translation_aug_1.augment(train['text'][i])
        bt2=back_translation_aug_2.augment(bt1)
        print(bt2)
        aug_text_not_sexist.append(bt1)
        aug_text_not_sexist.append(bt2)

BT_sexist_df = pd.DataFrame(aug_text_sexist)
BT_not_df = pd.DataFrame(aug_text_not_sexist)

BT_sexist_df.to_csv('/home/sanala/Juputer try/final_EDOS/augmented_data_bt_sexist.csv', index=False)
BT_not_df.to_csv('/home/sanala/Juputer try/final_EDOS/augmented_data_bt_not.csv', index=False)

