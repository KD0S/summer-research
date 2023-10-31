import pandas as pd

df = pd.read_csv('./data/aug_data_without_insert.csv')

df['TEXT'] = df['TEXT'].apply(lambda x: x[2:-2])
    
df.to_csv('./data/aug_data_without_insert_clean.csv')
    

# lengths = []

# # max_len = 0
# # max_words = 0

# # df = pd.read_csv('./data/dataset.csv')
# # for text in df['TEXT']:
# #     if len(text.split()) > max_words:
# #         max_words = len(text.split())
# #     if len(text) > max_len:
# #         max_len = len(text)

# df = df.drop(['SOURCE', 'Unnamed: 0', 'DECISION'], axis=1)
# print(df.head())

# df = pd.read_csv('./data/dataset2.csv')

# COUNT = 0
# for i, row in df.iterrows():
#     unique_id = i
#     text = row['IMPORTANT']
#     COUNT+=text
    

# # df.to_csv('./data/dataset2.csv')

# print(df.shape)
# print(COUNT)


# df = pd.read_csv('./data/cyberbully_tweets.csv')
# y = df['cyberbullying_type'].unique()
# print(y)
# dict = {}
# for i, y in enumerate(y):
#     dict[y]=i
    
# for id, row in df.iterrows():
#     row['cyberbullying_type'] = dict[row['cyberbullying_type']]

# y = df['cyberbullying_type']
# import torch
# labels = torch.nn.functional.one_hot(torch.tensor(y), num_classes=6)

# text = df['tweet_text'].values

