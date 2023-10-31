import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import text_hammer as th



# df_test = pd.read_csv('./data/test_data.csv')

# df.to_csv('./data/test_data.csv', mode='a', header=False)

# df = pd.read_csv('./data/dataset.csv')
# # x_train, x_test, y_train, y_test = train_test_split(df["TEXT"], df["IMPORTANT"], shuffle=True, stratify=df["IMPORTANT"],
# # #                                         test_size=0.157)
# seed_imp = []
# sentences = []

# for i, row in df.iterrows():
#     if row['SEED'] == 'IMP' or row['SEED'] == 'imp' or row['SEED'] == 'I':
#         sentences.append(row['TEXT'])
#         seed_imp.append(row['IMPORTANT'])

# for s in sentences:
#     df.drop(df.index[df['TEXT'] == s], inplace=True)
# df.to_csv('./data/dataset_final.csv')           
        
# df = pd.DataFrame({'TEXT':sentences, 'IMPORTANT': seed_imp})
# df.to_csv('./data/seeded_sentences.csv')

# df_train.to_csv('./data/train_set_final.csv')
# df_test.to_csv('./data/test_set_final.csv')

# df = pd.read_csv('./data/dataset_final.csv')

# x_train, x_test, y_train, y_test = train_test_split(df["TEXT"], df["IMPORTANT"], shuffle=True, stratify=df["IMPORTANT"],
#                                   test_size=0.157)
# df_train = pd.DataFrame({"TEXT": x_train, "IMPORTANT": y_train})
# df_test = pd.DataFrame({"TEXT": x_test, "IMPORTANT": y_test})

# df_train.to_csv('./data/train_data.csv')
# df_test.to_csv('./data/test_data.csv')

# df = pd.read_csv('./data/aug_datasets/aug_data_80_protected_words.csv')
# def text_preprocessing(df,col_name):
#     column = col_name
#     # df[column] = df[column].apply(lambda x:str(x).lower())
#     # df[column] = df[column].apply(lambda x: th.cont_exp(x)) 
#     # df[column] = df[column].apply(lambda x: th.remove_special_chars(x))
#     # df[column] = df[column].apply(lambda x: th.remove_accented_chars(x))
#     df[column] = df[column].apply(lambda x: x[2:-2]) 
#     return(df)

# df = text_preprocessing(df, 'TEXT')
# df.to_csv('./data/aug_datasets/aug_data_80_protected_words.csv')
# df.to_csv('./data/aug_datasets/aug_data_20_protected_words.csv', mode='a', header=False)


# df = text_preprocessing(df, 'TEXT')
# df.to_csv('./data/dataset.csv')

# df = pd.read_csv('./data/test_data.csv')
# df1 = pd.read_csv('./data/seeded_sentences.csv')

# # df = text_preprocessing(df, 'TEXT')
# df.to_csv('./data/aug_datasets/aug_data_20.csv', mode='a', header=False)
# df1.to_csv('./data/aug_datasets/aug_data_20.csv', mode='a', header=False)

# df = pd.read_csv('./data/aug_datasets/aug_data_60_protected_words.csv')
# df.to_csv('./data/aug_datasets/aug_data_80_protected_words.csv', mode='a', header=False)
# df1.to_csv('./data/aug_datasets/aug_data_20_protected_words.csv', mode='a', header=False)



# df = pd.read_csv('./data/dataset.csv')
# df.to_csv('./data/aug_data_80.csv', mode='a', header=False)


df = pd.read_csv('./data/dataset.csv')

sentences = []

total_characters = 0

for i, row in df.iterrows():
    sentences.append(row['TEXT'])
    
for s in sentences:
    total_characters+=len(s)
    
print(len(sentences), total_characters)