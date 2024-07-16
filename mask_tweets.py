import pandas as pd 
import re

tweets = pd.read_csv("all_tweets.csv")

for i in range(len(tweets['text'])):
    tweets.loc[i, 'text'] = re.sub(r'@\S+', '@user', tweets['text'][i])

tweets.to_csv('masked_all_tweets.csv', index=False)