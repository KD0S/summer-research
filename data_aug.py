import pandas as pd
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
from sklearn.model_selection import train_test_split

from nlpaug.util import Action

df = pd.read_csv('./data/aug_datasets/aug_data_80.csv')
x_train, x_test, y_train, y_test = train_test_split(df["TEXT"], df["IMPORTANT"], shuffle=True, stratify=df["IMPORTANT"],
                                        test_size=0.2)
protected_words = pd.read_csv('./data/ProtectedWords.csv')
protected_words = protected_words['WORDS'].values
words = []
for word in protected_words:
    words.append(word)
df = pd.DataFrame({"TEXT": x_test, "IMPORTANT": y_test})
text = df['TEXT'].values
important = df['IMPORTANT'].values

 
aug_inLegalBERT_sub = naw.ContextualWordEmbsAug(model_path='law-ai/InLegalBERT', action='substitute', device='cuda')
aug_BERTBase_sub = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action='substitute', device='cuda')
aug_legalBERT_sub = naw.ContextualWordEmbsAug(model_path='nlpaueb/legal-bert-base-uncased', action='substitute', device='cuda')
aug_incaselawbert_sub = naw.ContextualWordEmbsAug(model_path='law-ai/InCaseLawBERT', action='substitute', device='cuda')
aug_casebert_sub = naw.ContextualWordEmbsAug(model_path='casehold/legalbert', action='substitute', device='cuda')


aug_text = []
aug_important = []
count = 1

for row, i in zip(text, important):
    print(count)
    count+=1
    aug_text.append(aug_inLegalBERT_sub.augment(row))
    aug_important.append(i)
    aug_text.append(aug_legalBERT_sub.augment(row))
    aug_important.append(i)
    aug_text.append(aug_BERTBase_sub.augment(row))
    aug_important.append(i)
    aug_text.append(aug_incaselawbert_sub.augment(row))
    aug_important.append(i)
    aug_text.append(aug_casebert_sub.augment(row))
    aug_important.append(i)


    
aug_df = pd.DataFrame({"TEXT":aug_text, "IMPORTANT":aug_important})    
aug_df.to_csv('./data/aug_datasets/aug_data_100.csv')

