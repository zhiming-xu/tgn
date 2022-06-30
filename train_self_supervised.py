import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path

# xzl
from torch.profiler import profile, record_function, ProfilerActivity

from evaluation.evaluation import eval_edge_prediction
from model.tgn import TGN
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
from utils.data_processing import get_data, compute_time_statistics

# xzl
def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cpu_time_total", row_limit=10)
    print(output)
    p.export_chrome_trace("./traces/trace_" + str(p.step_num) + ".json")    

torch.manual_seed(0)
np.random.seed(0)

### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--bs', type=int, default=200, help='Batch_size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                  'backprop')
parser.add_argument('--use_memory', action='store_true',
                    help='Whether to augment the model with a node memory')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
    "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=[
    "mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=[
    "gru", "rnn"], help='Type of memory updater')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message '
                                                                        'aggregator')
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for '
                                                                'each user')
parser.add_argument('--different_new_nodes', action='store_true',
                    help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--dyrep', action='store_true',
                    help='Whether to run the dyrep model')

# xzl 
parser.add_argument('--inference_only', action='store_true',
                    help='xzl:do infer only, load a trained model')
parser.add_argument('--not_load_mem', action='store_true',
                    help='xzl:when load a trained model, not loading the memory state')
parser.add_argument('--train_split', type=float, default=0.7, help='train split. validation fixed 0.15. remaining for testing')
parser.add_argument('--mem_node_prob', type=float, default=1.0, help='%% of nodes that will have memory. default 1.0')
parser.add_argument('--fixed_edge_feature', action='store_true', default=False,
                    help="xzl:use fixed edge feature. the feature is the same as the source node's first edge")
parser.add_argument('--use_fixed_times', action='store_true', default=False,
                    help="xzl:use fixed timestamps sent to time encodings. the timestamp is cal as the avg of all ts in that batch")

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim
# --- below xzl ---- #
INFERENCE_ONLY = args.inference_only   
TRAIN_SPLIT = args.train_split  

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}.pth'
get_checkpoint_path = lambda \
    epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO) # xzl
Path("log/").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

### Extract data for training, validation and testing
# xzl)@new_node_val/test are nodes never showed up in training.
node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
new_node_test_data = get_data(DATA,
                              different_new_nodes_between_val_and_test=args.different_new_nodes, randomize_features=args.randomize_features,
                              train_split=TRAIN_SPLIT, fixed_edge_feat=args.fixed_edge_feature)

# Initialize training neighbor finder to retrieve temporal graph
train_ngh_finder = get_neighbor_finder(train_data, args.uniform)

# Initialize validation and test neighbor finder to retrieve temporal graph
full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

# Initialize negative samplers. Set seeds for validation and testing so negatives are the same
# across different runs
# NB: in the inductive setting, negatives are sampled only amongst other new nodes
train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations,
                                      seed=1)
test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources,
                                       new_node_test_data.destinations,
                                       seed=3)

# Set device
if GPU < 0: # xzl  much faster!! 
    print("xzl: force using cpu")
    device_string = 'cpu'  
    torch.set_num_threads(20)
    torch.set_num_interop_threads(20)
else:
    device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

# Compute time statistics   xzl: needed by tgn model --- to normalized time diff for encoding
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
    compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

for i in range(args.n_runs):
    results_path = "results/{}_{}.pkl".format(args.prefix, i) if i > 0 else "results/{}.pkl".format(args.prefix)
    Path("results/").mkdir(parents=True, exist_ok=True)

    # Initialize Model
    tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
              edge_features=edge_features, device=device,
              n_layers=NUM_LAYER,
              n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
              message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
              memory_update_at_start=not args.memory_update_at_end,
              embedding_module_type=args.embedding_module,
              message_function=args.message_function,
              aggregator_type=args.aggregator,
              memory_updater_type=args.memory_updater,
              n_neighbors=NUM_NEIGHBORS,
              mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
              mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
              use_destination_embedding_in_message=args.use_destination_embedding_in_message,
              use_source_embedding_in_message=args.use_source_embedding_in_message,
              dyrep=args.dyrep,
              mem_node_prob=args.mem_node_prob, use_fixed_times=args.use_fixed_times)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(tgn.parameters(), lr=LEARNING_RATE)
    tgn = tgn.to(device)

    if not INFERENCE_ONLY: 
        num_instance = len(train_data.sources) # xzl: instances == # of rows in training? 
        num_batch = math.ceil(num_instance / BATCH_SIZE)

        logger.info('train split: {}'.format(TRAIN_SPLIT))
        logger.info('num of training instances: {}'.format(num_instance))
        logger.info('num of batches per epoch: {}'.format(num_batch))
        idx_list = np.arange(num_instance)

        new_nodes_val_aps = []
        val_aps = []
        epoch_times = []
        total_epoch_times = []
        train_losses = []

        early_stopper = EarlyStopMonitor(max_round=args.patience)
        for epoch in range(NUM_EPOCH):
            start_epoch = time.time()
            ### Training

            # Reinitialize memory of the model at the start of each epoch
            if USE_MEMORY:
                tgn.memory.__init_memory__()

            # Train using only training graph
            tgn.set_neighbor_finder(train_ngh_finder)
            m_loss = []

            logger.info('start {} epoch'.format(epoch))
            with profile(
                #activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
              activities=[],
              schedule=torch.profiler.schedule(
                  wait=1,
                warmup=1,
                active=2),
              on_trace_ready=trace_handler,
              with_stack=True,
            ) as p:
                for k in range(0, num_batch, args.backprop_every):
                    loss = 0
                    optimizer.zero_grad()

                    # Custom loop to allow to perform backpropagation only every a certain number of batches
                    for j in range(args.backprop_every):
                        batch_idx = k + j

                        if batch_idx >= num_batch:
                            continue

                        start_idx = batch_idx * BATCH_SIZE
                        end_idx = min(num_instance, start_idx + BATCH_SIZE)
                        sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], \
                                                            train_data.destinations[start_idx:end_idx]
                        edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
                        timestamps_batch = train_data.timestamps[start_idx:end_idx]

                        size = len(sources_batch)
                        # xzl: how to ensure these edges are neg?  (sampled sources seem discarded)
                        _, negatives_batch = train_rand_sampler.sample(size) 

                        with torch.no_grad():
                            pos_label = torch.ones(size, dtype=torch.float, device=device)
                            neg_label = torch.zeros(size, dtype=torch.float, device=device)

                        tgn = tgn.train() # xzl: mark the start of training
                        # xzl: @negatives_batch are neg dest. 
                        #print("xzl: timestamps for the batch", timestamps_batch)

                        pos_prob, neg_prob = tgn.compute_edge_probabilities(sources_batch, destinations_batch, negatives_batch,
                                                                            timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS)

                        # xzl: @loss is a tensor. @criterion is a loss func (BCEloss)
                        loss += criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)

                    loss /= args.backprop_every

                    loss.backward()
                    optimizer.step()
                    m_loss.append(loss.item())

                    # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
                    # the start of time   
                    # xzl) shallow backprop (in time) ... to avoid excessive memory usage (gradients etc)
                    #        
                    if USE_MEMORY:
                        tgn.memory.detach_memory()

                    if k % 100 == 0:
                        print("# batch =", k)
                        #p.step() 

            epoch_time = time.time() - start_epoch
            epoch_times.append(epoch_time)

            ### Validation
            # Validation uses the full graph  (xzl: why not the only validation data??)
            tgn.set_neighbor_finder(full_ngh_finder)

            if USE_MEMORY:
                # Backup memory at the end of training, so later we can restore it and use it for the
                # validation on unseen nodes
                train_memory_backup = tgn.memory.backup_memory()

            # xzl: instead of predicting on all possible links (node pairs), only test on: pos edges (ground truth) and 
            #   sampled neg edges.     in theory: should test on all neg links. but too many?
            val_auc = eval_edge_prediction(model=tgn, negative_edge_sampler=val_rand_sampler,
                                           data=val_data, n_neighbors=NUM_NEIGHBORS)

            if USE_MEMORY:
                val_memory_backup = tgn.memory.backup_memory()
                # Restore memory we had at the end of training to be used when validating on new nodes.
                # Also backup memory after validation so it can be used for testing (since test edges are
                # strictly later in time than validation edges)
                tgn.memory.restore_memory(train_memory_backup)

            # Validate on unseen nodes (xzl: nn=new nodes, why this?)
            nn_val_auc = eval_edge_prediction(model=tgn, negative_edge_sampler=val_rand_sampler,
                                              data=new_node_val_data, n_neighbors=NUM_NEIGHBORS)

            if USE_MEMORY:
                # Restore memory we had at the end of validation
                tgn.memory.restore_memory(val_memory_backup)

            train_losses.append(np.mean(m_loss))

            # Save temporary results to disk
            pickle.dump({
                "val_aps": val_aps,
              "new_nodes_val_aps": new_nodes_val_aps,
              "train_losses": train_losses,
              "epoch_times": epoch_times,
              "total_epoch_times": total_epoch_times
            }, open(results_path, "wb"))

            total_epoch_time = time.time() - start_epoch
            total_epoch_times.append(total_epoch_time)

            logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
            logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
            logger.info(
                'val auc: {}, new node val auc: {}'.format(val_auc, nn_val_auc))

            # Early stopping
            if early_stopper.early_stop_check(val_auc):
                logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
                logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
                best_model_path = get_checkpoint_path(early_stopper.best_epoch)
                tgn.load_state_dict(torch.load(best_model_path))
                logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
                tgn.eval()
                break
            else:
                torch.save(tgn.state_dict(), get_checkpoint_path(epoch))
    else:   # xzl: INFERENCE_ONLY. the model must exist
        # tgn.load_state_dict(torch.load(MODEL_SAVE_PATH)) 
        if args.not_load_mem:
            tgn.memory.clear_memory()  

    # Training has finished, we have loaded the best model, and we want to backup its current
    # memory (which has seen validation edges) so that it can also be used when testing on unseen
    # nodes
    # xzl: this backs up memory & msg stores. 
    #   @test_data contains both old and new nodes (i.e. unseen in training), while @new_node_test_data
    #   has only new nodes. reset memory to the end of training --> ensures no msgs/memory about the unseen nodes
    if USE_MEMORY:
        val_memory_backup = tgn.memory.backup_memory()

    ### Test
    tgn.embedding_module.neighbor_finder = full_ngh_finder
    test_auc = eval_edge_prediction(model=tgn, negative_edge_sampler=test_rand_sampler,
                                    data=test_data, n_neighbors=NUM_NEIGHBORS)

    if USE_MEMORY:
        tgn.memory.restore_memory(val_memory_backup)

    # Test on unseen nodes
    nn_test_auc = eval_edge_prediction(model=tgn, negative_edge_sampler=nn_test_rand_sampler,
                                                   data=new_node_test_data, n_neighbors=NUM_NEIGHBORS)

    logger.info(
        'Test statistics: Old nodes -- auc: {}, ap: {}'.format(test_auc))
    logger.info(
        'Test statistics: New nodes -- auc: {}, ap: {}'.format(nn_test_auc))

    if INFERENCE_ONLY: 
        pickle.dump({
            "test_auc": test_auc,
          "new_node_test_auc": nn_test_auc,
        }, open(results_path, "wb"))
        # save memory as file, for inspection
        mem, last_update, msgs = tgn.memory.backup_memory()
        path = './data/inference-only-memory.npy'
        np.save(path, np.array(mem.cpu()))
        logger.info('Inference done. mem saved to {}'.format(path))

    else: 
        # Save results for this run
        pickle.dump({
            "val_aps": val_aps,
          "new_nodes_val_aps": new_nodes_val_aps,
          "test_auc": test_auc,
          "new_node_test_auc": nn_test_auc,
          "epoch_times": epoch_times,
          "train_losses": train_losses,
          "total_epoch_times": total_epoch_times
        }, open(results_path, "wb"))

        logger.info('Saving TGN model')
        if USE_MEMORY:
            # Restore memory at the end of validation (save a model which is ready for testing)
            tgn.memory.restore_memory(val_memory_backup)
        torch.save(tgn.state_dict(), MODEL_SAVE_PATH)
        logger.info('TGN model saved')
