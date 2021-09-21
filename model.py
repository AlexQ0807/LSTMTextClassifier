import numpy as np
import torch
import torch.nn as nn
from wordenums import UNK

class Model(nn.Module):
    def __init__(self, dataset, embedding_dim, num_layers, hidden_size, output_size):
        super(Model, self).__init__()
        self.dataset = dataset
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # Embedding layer converts word indexes to word vectors
        n_vocab = len(self.dataset.word_to_index.keys())

        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim,
        )

        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

        self.fc = nn.Linear(self.hidden_size, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, prev_state):
        batch_size = x.shape[0]

        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)  # (batch, seq, input)
        output = output.contiguous().view(-1, self.hidden_size)  # (batch*seq, input)
        logits = self.fc(output)

        # sigmoid function
        sig_out = self.sig(logits)  # (batch*seq, 1)
        sig_out = sig_out.view(batch_size, -1)  # (batch, seq)
        sig_out = sig_out[:, -1]  # (batch) - take output of the last batch

        return sig_out, state

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))

    def make_text_prediction(self, text):
        # Preprocess the text
        text = self.dataset.preprocess_text(text)
        text = text.split(" ")

        # Convert words to index numbers
        tokens = []
        for word in text:
            if word in self.dataset.word_to_index:
                tokens.append(self.dataset.word_to_index[word])
            else:
                tokens.append(self.dataset.word_to_index[UNK])

        # Pass processed to the model and obtain the prediction result
        inp = torch.from_numpy(np.array(tokens))
        hidden = self.init_hidden(1)
        out, _ = self.forward(torch.unsqueeze(inp, 0), hidden)

        pred_index = int(torch.round(out.squeeze()).item())
        pred_label = self.dataset.index_to_label[pred_index]

        return pred_label

    def save_model(self, save_model_filepath):
        torch.save(self.state_dict(), save_model_filepath)
        print("Saved model to {}".format(save_model_filepath))