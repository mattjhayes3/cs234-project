import pandas as pd
import random

threshold = 0.25
uniform = False

random.seed(42)

input = "pref_pairs_16_token_2_choose_2.csv"

df = pd.read_csv(input, index_col=0)

print(df.columns)

# flips = (df['diff'] < threshold) & (df.index < len(df)/1.5)
# chosen = df.loc[flips, 'chosen']
# rejected = df.loc[flips, 'rejected']
quantile = df['diff'].quantile(threshold)

count = 0
for index, row in df.iterrows():
    if uniform:
        cond = random.random() < threshold
    else: 
        cond = row['diff'] < quantile
    if cond:
        print('flipping diff', row['diff'])
        count += 1
        chosen = row['chosen']
        rejected = row['rejected']
        df.loc[index, 'chosen'] = rejected
        df.loc[index, 'rejected'] = chosen
print(f"Flipped {count}/{len(df)}")
df.to_csv(f"{input[:-4]}_{'uniform' if uniform else 'quantile'}_flipped_{str(threshold).replace('.', '_')}.csv")