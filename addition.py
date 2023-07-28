import numpy as np
import torch.utils.data
import simple_transformer
import torch.nn as nn
import torch.nn.functional as F
import torch
from tqdm import tqdm


MAX_LENGTH = 16

vocabulary = {ch: i for i, ch in enumerate(' 0123456789+-*/=')}


def generate_symetric(min_value, max_value):
    if np.random.rand() > 0.5:
        return np.random.randint(min_value, max_value)
    else:
        return -np.random.randint(min_value, max_value)
    pass

def generate_record(min_value, max_value):
    # generate left operand
    left = generate_symetric(min_value, max_value)
    # generate right operand
    right = generate_symetric(min_value, max_value)
    # generate answer
    answer = left + right

    # generate string
    query_str = f'{left} + {right}'
    answer_str = str(answer)

    return list(query_str), list(answer_str)


def vectorize_record(input: list):
    vector = np.zeros(len(input))
    for i, ch in enumerate(input):
        vector[i] = vocabulary[ch]
        pass
    return vector


class AdditionDataset(torch.utils.data.Dataset):
    def __init__(self, min_value, max_value, size):
        self.min_value = min_value
        self.max_value = max_value
        self.size = size
        pass

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        query, answer = generate_record(self.min_value, self.max_value)
        query.reverse()
        answer.reverse()
        query_vector = vectorize_record(query)
        answer_vector = vectorize_record(answer)
        return query_vector, answer_vector
    pass


def collate_fun(batch):
    # batch is a list of tuples (query, answer)
    batch_size = len(batch)

    x = torch.zeros((batch_size, MAX_LENGTH), dtype=torch.long)
    y = torch.zeros((batch_size, MAX_LENGTH), dtype=torch.long)

    for i, (q, a) in enumerate(batch):
        x[i, :len(q)] = torch.from_numpy(q).long()
        y[i, :len(a)] = torch.from_numpy(a).long()
        pass

    return x, y
    pass


class AdditionModel(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.attention_head_size = 8
        self.num_attention_heads = hidden_size // self.attention_head_size
        self.encoder = simple_transformer.MultiLayerTransformer(
            num_layers = 4,
            hidden_size=hidden_size,
            attention_head_size=self.attention_head_size,
            num_attention_heads=4,
            reduction=None,
            use_structure_matrix=True
        )

        self.relative_position_embedding = nn.Embedding(2 * MAX_LENGTH - 1, self.attention_head_size)
        self.embedding = nn.Embedding(len(vocabulary), hidden_size)
        
        seq = torch.arange(MAX_LENGTH)
        relative_position = seq[:, None] - seq[None, :]
        relative_position += MAX_LENGTH - 1

        self.register_buffer('relative_position', relative_position)
        self.classifier = nn.Linear(hidden_size, len(vocabulary))
        pass

    def forward(self, x):
        # x is a batch of query vectors
        batch_size = x.shape[0]
        x = self.embedding(x)
        pos_emb = self.relative_position_embedding(self.relative_position)
        x = self.encoder(x, structure_matrix=pos_emb)
        x = self.classifier(x)
        return x
        pass
    pass


def train():

    # define dataset
    dataset_train = AdditionDataset(0, 1000, 10000)
    dataset_test = AdditionDataset(1000, 10000, 10000)

    # define data loader
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=1024, shuffle=True, collate_fn=collate_fun)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1024, shuffle=True, collate_fn=collate_fun)
    
    # define model
    model = AdditionModel(32)
    model.to('mps')
    # define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # train loop

    for epoch in range(1000):
        pbar = tqdm(total=len(data_loader_train))
        for x, y in data_loader_train:
            x = x.to('mps')
            y = y.to('mps')
            optimizer.zero_grad()
            y_pred = model(x.long())
            loss = F.cross_entropy(y_pred.transpose(1, 2), y.long())
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': loss.item()})
            pbar.update(1)
            pass

        pbar.close()

        # test
        test_losses = []
        with torch.no_grad():
            pbar = tqdm(total=len(data_loader_test))
            for x, y in data_loader_test:
                x = x.to('mps')
                y = y.to('mps')
                y_pred = model(x.long())
                loss = F.cross_entropy(y_pred.transpose(1, 2), y.long())
                test_losses.append(loss.item())
                pbar.set_postfix({'loss': loss.item()})
                pbar.update(1)
                pass
            pass
        pbar.close()
        print(f'Epoch {epoch} test loss: {np.mean(test_losses)}')
        pass
    pass


if __name__ == '__main__':
    train()