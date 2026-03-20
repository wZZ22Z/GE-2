import numpy as np
import torch

from gege.tools.preprocess.converters.partitioners.partitioner import Partitioner


def dataframe_to_tensor(input_dataframe):
    np_array = input_dataframe.to_dask_array().compute()
    return torch.from_numpy(np_array)


def partition_edges(edges, num_nodes, num_partitions):
    partition_size = int(np.ceil(num_nodes / num_partitions))

    # All node ids are non-negative, so floor division is equivalent to truncation
    # and works across older Torch versions that do not support rounding_mode.
    src_partitions = edges[:, 0] // partition_size
    dst_partitions = edges[:, -1] // partition_size

    # Sort by a single flat bucket id instead of relying on stable multi-pass sort.
    edge_bucket_ids = src_partitions * num_partitions + dst_partitions
    edge_bucket_ids, sort_args = torch.sort(edge_bucket_ids)
    edges = edges.index_select(0, sort_args)

    offsets = np.zeros(num_partitions * num_partitions, dtype=int)

    unique_buckets, bucket_counts = torch.unique_consecutive(edge_bucket_ids, return_counts=True)
    offsets[unique_buckets.cpu().numpy()] = bucket_counts.cpu().numpy()
    offsets = list(offsets)

    return edges, offsets


class TorchPartitioner(Partitioner):
    def __init__(self, partitioned_evaluation):
        super().__init__()

        self.partitioned_evaluation = partitioned_evaluation

    def partition_edges(self, train_edges_tens, valid_edges_tens, test_edges_tens, num_nodes, num_partitions):
        """ """

        train_edges_tens, train_offsets = partition_edges(train_edges_tens, num_nodes, num_partitions)

        valid_offsets = None
        test_offsets = None

        if self.partitioned_evaluation:
            if valid_edges_tens is not None:
                valid_edges_tens, valid_offsets = partition_edges(valid_edges_tens, num_nodes, num_partitions)

            if test_edges_tens is not None:
                test_edges_tens, test_offsets = partition_edges(test_edges_tens, num_nodes, num_partitions)

        return train_edges_tens, train_offsets, valid_edges_tens, valid_offsets, test_edges_tens, test_offsets
