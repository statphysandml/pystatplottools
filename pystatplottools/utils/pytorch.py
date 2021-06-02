

# Optional function for converting given output from the train_loader to a corresponding batch
def batch_converter(batch):
    if isinstance(batch, list) or isinstance(batch, tuple):
        batch_y = batch[1]
        batch = batch[0]
    else:
        batch_y = batch.y
    return batch, batch_y