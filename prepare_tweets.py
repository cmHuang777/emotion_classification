import pandas as pd

tweets = pd.read_csv("masked_all_tweets.csv")
qc_tweets = tweets.iloc[:300]
other_tweets = tweets.iloc[300:]

# split the tweets into 4 sets each containing around 600 tweets
# of which first 75 rows will be from qc_tweets

for i in range(3):
    qc_tweets_split = qc_tweets.iloc[i*75:(i+1)*75]
    other_tweets_split = other_tweets.iloc[i*525:(i+1)*525]
    df = pd.concat((qc_tweets_split, other_tweets_split))
    df.to_csv(f'tweets_{i}.csv', index=False)

# last split has more rows
qc_tweets_split = qc_tweets.iloc[225:]
other_tweets_split = other_tweets.iloc[1575:]
print("qc length: ", len(qc_tweets_split))
print("others length: ", len(other_tweets_split))

df = pd.concat((qc_tweets_split, other_tweets_split))
df.to_csv(f'tweets_{3}.csv', index=False)