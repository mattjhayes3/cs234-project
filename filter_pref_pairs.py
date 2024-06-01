import pandas as pd


input = "pref_pairs_16_token_2_choose_2.csv"
threshold = 0.3

df = pd.read_csv(input, index_col=0)

filtered = df.loc[df['diff'] > threshold]
print(f"Filtered {len(filtered)}/{len(df)}")
filtered.to_csv(f"{input[:-4]}_filtered_{str(threshold).replace('.', '_')}.csv")