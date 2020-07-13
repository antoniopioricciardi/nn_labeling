from torch.utils.data import DataLoader
import torch

class Trainer:

    def generate_batch(self, batch):
        label = torch.tensor([entry[0] for entry in batch])  # take labels from batch and convert them to tensors
        text = [entry[1] for entry in batch]
        print(batch)
        exit(2)

        # this should create a list of lengths, with 0-length in first position (due to [0] + ..)
        offsets = [0] + [len(entry) for entry in text]
        # torch.Tensor.cumsum returns the cumulative sum
        # of elements in the dimension dim.
        # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)

        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text = torch.cat(text)
        return text, offsets, label

    def train(self, train_data, BATCH_SIZE):
        # Train the model
        train_loss = 0
        train_acc = 0
        data = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=self.generate_batch)
        for i, (text, offsets, cls) in enumerate(data):
            print(text, offsets, cls)
        #     optimizer.zero_grad()
        #     text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        #     output = model(text, offsets)
        #     loss = criterion(output, cls)
        #     train_loss += loss.item()
        #     loss.backward()
        #     optimizer.step()
        #     train_acc += (output.argmax(1) == cls).sum().item()
        #
        # # Adjust the learning rate
        # scheduler.step()
        #
        # return train_loss / len(sub_train_), train_acc / len(sub_train_)
