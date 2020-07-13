import torch


def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])  # take labels from batch and convert them to tensors
    text = [entry[1] for entry in batch]
    # this should create a list of lengths, with 0-length in first position (due to [0] + ..)
    offsets = [0] + [len(entry) for entry in text]
    # torch.Tensor.cumsum returns the cumulative sum
    # of elements in the dimension dim.
    # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)

    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label
