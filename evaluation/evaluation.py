import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def eval_edge_prediction(model, negative_edge_sampler, data, n_neighbors):
    # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
    # negatives for validation / test set)
    assert negative_edge_sampler.seed is not None
    negative_edge_sampler.reset_random_state()

    val_auc = []
    with torch.no_grad():
        model = model.eval()
        # While usually the test batch size is as big as it fits in memory, here we keep it the same
        # size as the training batch size, since it allows the memory to be updated more frequently,
        # and later test batches to access information from interactions in previous test batches
        # through the memory
        num_test_instance = len(data.sources)
        # candidates = data.destinations
        sorted_timestamps = np.unique(data.timestamps)
        sorted_timestamps.sort()

        # xzl: create batches and send through prediction (@compute_edge_probabilities, encoding and then decoding)
        # for each timestamp, first predict the edge probabilities
        for ts in sorted_timestamps:
            index = data.timestamps == ts
            sources_batch = data.sources[index]
            destinations_batch = data.destinations[index]
            edge_time_batch = data.timestamps[index]
            edge_idxs_batch = data.edge_idxs[index]
            auc_scores = []
            for source in np.unique(sources_batch):
                src_index = (sources_batch == source)
                # extract all nodes happen before the end time, incl. new nodes
                candidates = np.unique(data.destinations[data.timestamps<=ts])
                pred_score = model.compute_all_pairs_probabilities(source, ts, candidates, n_neighbors)
                real_dst = destinations_batch[src_index]
                true_label = torch.tensor([c in real_dst for c in candidates], dtype=torch.long)
                try:
                    auc_scores.append(roc_auc_score(true_label.cpu(), pred_score.cpu()))
                except:
                    # skip those with too few candidates
                    pass

        # for each timestamp, then update the memory
            model.compute_embedding_and_update_memory(sources_batch, destinations_batch, edge_time_batch, edge_idxs_batch)
            # auc is not very meaningful now
            # val_ap.append(average_precision_score(true_label, pred_score))
            val_auc.extend(auc_scores)

    return np.mean(val_auc)


def eval_node_classification(tgn, decoder, data, edge_idxs, batch_size, n_neighbors):
    pred_prob = np.zeros(len(data.sources))
    num_instance = len(data.sources)
    num_batch = math.ceil(num_instance / batch_size)

    with torch.no_grad():
        decoder.eval()
        tgn.eval()
        for k in range(num_batch):
            s_idx = k * batch_size
            e_idx = min(num_instance, s_idx + batch_size)

            sources_batch = data.sources[s_idx: e_idx]
            destinations_batch = data.destinations[s_idx: e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_idxs_batch = edge_idxs[s_idx: e_idx]

            source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
                                                                                         destinations_batch,
                                                                                         destinations_batch,
                                                                                         timestamps_batch,
                                                                                         edge_idxs_batch,
                                                                                         n_neighbors)
            pred_prob_batch = decoder(source_embedding).sigmoid()
            pred_prob[s_idx: e_idx] = pred_prob_batch.cpu().numpy()

    auc_roc = roc_auc_score(data.labels, pred_prob)
    return auc_roc
