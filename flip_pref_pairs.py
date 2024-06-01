import pandas as pd
import random

threshold = 0.2

random.seed(42)

input = "pref_pairs_16_token_2_choose_2.csv"

df = pd.read_csv(input, index_col=0)

print(df.columns)

# flips = (df['diff'] < threshold) & (df.index < len(df)/1.5)
# chosen = df.loc[flips, 'chosen']
# rejected = df.loc[flips, 'rejected']
count = 0
for index, row in df.iterrows():
    if random.random() > row['diff'] + threshold:
        count += 1
        chosen = row['chosen']
        rejected = row['rejected']
        df.loc[index, 'chosen'] = rejected
        df.loc[index, 'rejected'] = chosen
print(f"Flipped {count}/{len(df)}")
df.to_csv(f"{input[:-4]}_flipped_{str(threshold).replace('.', '_')}.csv")