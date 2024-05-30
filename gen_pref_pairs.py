from itertools import combinations 
import pandas as pd

DISTINCT_GENERATIONS = 6
OUTFILE = 'pref_pairs_16_token_6_choose_2_div_6.csv'
TOTAL = 24895
DISTINCT_PAIRS = False

# dfs = [pd.read_csv(f'sft_generations_v2_{i}.csv', index_col=0) for i in range(5)]
# df = pd.concat(dfs, axis=0).reset_index(drop=True)
df = pd.read_csv('sft_generations_16_token.csv')
# df.to_csv('sft_generations_v2_merged.csv')
print("total len", len(df))

pref_pairs = []
dupes = 0
for index, row in df.iterrows():
    generations = set()
    gen_sents = []
    for i in range(8):
        gen_sent = (row[f'generation_{i}'], row[f'sentiment_{i}'])
        if gen_sent[0] in generations:
            continue
        gen_sents.append(gen_sent)
        generations.add(gen_sent[0])
        if len(gen_sents) == DISTINCT_GENERATIONS:
            break
    if not len(gen_sents) == DISTINCT_GENERATIONS:
        print("skipping index", index)
        continue

    # gen_sents = [(row[f'generation_{i}'], row[f'sentiment_{i}']) for i in range(4)]

    generator = combinations(gen_sents, 2) 
    if DISTINCT_PAIRS:
        assert DISTINCT_GENERATIONS % 2 == 0
        generator = []
        for i in range(DISTINCT_GENERATIONS//2):
            generator.append((gen_sents[i*2], gen_sents[i*2 + 1]))
    for a, b in generator:
        if (a[0] == b[0]):
            dupes +=1
            continue
        winner, loser = (a, b) if a[1] > b[1] else (b, a)
        pref_pairs.append({
            'prompt': row.query,
            'chosen': winner[0],
            'rejected': loser[0],
            'diff': winner[1] - loser[1],
            'source_index': index,
        })
        if len(pref_pairs) == TOTAL:
            print(f"Stopping at index {index}")
            break
    if len(pref_pairs) == TOTAL:
        print(f"Stopping at index {index}")
        break
result = pd.DataFrame(pref_pairs)
result.to_csv(OUTFILE)
print("Generated", len(result), "dropped", dupes, "exact dupes")
