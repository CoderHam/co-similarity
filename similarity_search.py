import pandas as pd
import numpy as np
import faiss

np.random.seed(4)

# Load files with scraped and preprocessed sentences, embeddings
records = pd.read_csv("bgg_2000.csv")
sentence_df = pd.read_csv("bgg_2000_sentences.csv")
sentence_embeddings = np.load('data/all_embed.npy').astype(np.float32)

# Get KNN using faiss
# I is the index for the K nearest neighbors
# D is the distance metric for the K nearest neighbors
def run_knn(sentence_embeds, query_embeds, k=5):
    dims = sentence_embeddings.shape[1]

    # sentence_embeds[:, 0] += np.arange(sentence_embeds.shape[0]) / 1000.
    # query_embeds[:, 0] += np.arange(query_embeds.shape[0]) / 1000.

    index_flat = faiss.IndexFlatL2(dims)
    index_flat.add(sentence_embeds)
    D, I = index_flat.search(query_embeds, k)

    return I, D

# K defaults to 5
def get_similar(game_id=0, k=5, select_sentence=False, verbose=False):
    if verbose:
        print("%s\n-----\n" % records.loc[game_id, 'name'])
        query_sentences = sentence_df.loc[sentence_df['uid'] == game_id, 'sentence'].to_list()
        for i, qs in enumerate(query_sentences):
            print("%s: %s" % (i, qs))

    query_embeddings = np.load('data/%s.npy' % game_id).astype(np.float32)

    # if no sentence is selected then run for all sentences
    if select_sentence:
        query_id = int(input())
        many_indicies, many_dists = run_knn(sentence_embeddings, query_embeddings[query_id:query_id+1], k)
    else:
        many_indicies, many_dists = run_knn(sentence_embeddings, query_embeddings, k)

    results = []
    for indices in many_indicies:
        uids = sentence_df.loc[indices, 'uid'].to_list()

        # filter out uid of game being queried
        filtered_uids = list(filter(lambda uid: uid != game_id, uids))
        results.extend(filtered_uids)

    # TODO Take distance into account
    # TODO set does not preserve order
    return list(set(results))

# Demo of usage in jupyter notebook

# import timeit
# num_runs = 10
# duration = timeit.Timer(get_similar).timeit(number = num_runs)
# print(f'On average it took {duration*1000/num_runs} ms')
# 126 ms with K = 5