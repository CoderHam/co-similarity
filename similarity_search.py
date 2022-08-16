import pandas as pd
import numpy as np
import faiss

np.random.seed(4)

# Load files with scraped and preprocessed sentences, embeddings
records = pd.read_csv("bgg_2000.csv")
sentence_df = pd.read_csv("bgg_2000_sentences.csv")
sentence_embeddings = np.load('data/all_embed.npy').astype(np.float32)

# Create index (can also run on GPU)
def create_index(use_gpu=False):
    dims = sentence_embeddings.shape[1]

    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexIVFFlat(res, dims, 32, faiss.METRIC_L2)
    else:
        quantizer = faiss.IndexFlatL2(dims)
        index = faiss.IndexIVFFlat(quantizer, dims, 32, faiss.METRIC_L2)
    index.train(sentence_embeddings)
    index.add(sentence_embeddings)

    chunk = faiss.serialize_index(index)
    return chunk

# Load index (can also run on GPU)
# res = faiss.StandardGpuResources()
index = faiss.deserialize_index(create_index(use_gpu=False))
# index = faiss.index_cpu_to_gpu(res, 0, index)

# Get KNN using faiss
# I is the index for the K nearest neighbors
# D is the distance metric for the K nearest neighbors
def run_knn(query_embeds, k=5):
    D, I = index.search(query_embeds, k)
    return I, D

# K defaults to 5
def get_similar_from_game(game_id=0, k=5, select_sentence=False, verbose=False):
    if verbose:
        print("%s\n-----\n" % records.loc[game_id, 'name'])
        query_sentences = sentence_df.loc[sentence_df['uid'] == game_id, 'sentence'].to_list()
        for i, qs in enumerate(query_sentences):
            print("%s: %s" % (i, qs))

    query_embeddings = np.load('data/%s.npy' % game_id).astype(np.float32)

    # if no sentence is selected then run for all sentences
    if select_sentence:
        query_id = int(input())
        many_indicies, many_dists = run_knn(query_embeddings[query_id:query_id+1], k)
    else:
        many_indicies, many_dists = run_knn(query_embeddings, k)

    result_uids = []
    result_dists = []
    for indices, dists in zip(many_indicies, many_dists):
        uids = sentence_df.loc[indices, 'uid'].to_list()
        # filter out uid of game being queried for
        result_uids.extend([uid for uid in uids if uid!= game_id])
        result_dists.extend([dist for dist, uid in zip(dists, uids) if uid!= game_id])

    sorted_uids = [x for _, x in sorted(zip(result_dists, result_uids))]
    return sorted_uids[:k]

# Demo of usage in jupyter notebook
# import timeit
# num_runs = 10
# duration = timeit.Timer(get_similar_from_game).timeit(number = num_runs)
# print(f'On average it took {duration*1000/num_runs} ms')
# 6 ms with K = 5

def get_similar_from_embeds(query_embeddings=0, k=5):
    query_embeddings = np.array(query_embeddings).astype(np.float32)

    many_indicies, many_dists = run_knn(query_embeddings, k)

    result_uids = []
    result_dists = []
    for indices, dists in zip(many_indicies, many_dists):
        uids = sentence_df.loc[indices, 'uid'].to_list()
        result_uids.extend(uids)
        result_dists.extend(dists)

    sorted_uids = [x for _, x in sorted(zip(result_dists, result_uids))]
    return sorted_uids[:k]
