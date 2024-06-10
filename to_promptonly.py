import pandas as pd

df = pd.read_csv('./data/pref_pairs_16_token_4_choose_2_div_6.csv')

df.groupby('prompt').first().to_csv('./data/pref_pairs_16_token_4_choose_2_div_6_promptonly.csv')