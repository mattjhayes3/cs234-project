import pandas as pd


input = "pref_pairs_16_token_2_choose_2.csv"
threshold = 0.3

df = pd.read_csv(input, index_col=0)

print(df.columns)

flips = df['diff'] > threshold
chosen = df.loc[flips, 'chosen']
rejected = df.loc[flips, 'rejected']
df.loc[flips, 'chosen'] = rejected
df.loc[flips, 'rejected'] = chosen
print(f"Flipped {flips.sum()}/{len(df)}")
df.to_csv(f"{input[:-4]}_flipped_{str(threshold).replace('.', '_')}.csv")