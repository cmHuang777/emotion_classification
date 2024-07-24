import pandas as pd

# Read the labeled CSVs back into dataframes
df1 = pd.read_csv("data/drone/responses/batch1_tweets_full_responses.csv")
df2 = pd.read_csv("data/drone/responses/batch2_tweets_full_responses.csv")
df3 = pd.read_csv("data/drone/responses/batch3_tweets_full_responses.csv")
df4 = pd.read_csv("data/drone/responses/batch4_tweets_full_responses.csv")

# Concatenate the dataframes in the order they were split
combined_df = pd.concat([df1, df2, df3, df4], ignore_index=True)

# Restore the original split
qc_tweets_restored = pd.concat(
    [
        combined_df.iloc[:75],
        combined_df.iloc[600:675],
        combined_df.iloc[1200:1275],
        combined_df.iloc[1800:2100],
    ]
)

other_tweets_restored = pd.concat(
    [
        combined_df.iloc[75:600],
        combined_df.iloc[675:1200],
        combined_df.iloc[1275:1800],
        combined_df.iloc[2100:],
    ]
)

# Combine the restored parts
restored_df = pd.concat([qc_tweets_restored, other_tweets_restored], ignore_index=True)

# print(restored_df.describe())

# Save the restored dataframe to a new CSV file
restored_df.to_csv("data/drone/responses/all_tweets_full_responses.csv", index=False)
