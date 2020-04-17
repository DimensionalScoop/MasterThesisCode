from icae.tools.dataset_sparse import SparseEventDataset

dataset = SparseEventDataset(size=100)

sample_event = dataset._get_single_event(0).to_dense()
# sample_batch = 
shape = sample_event.shape
channels = shape[0]
aspect_ratio = shape[1] / shape[2]