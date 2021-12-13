import numpy as np
from collections import Counter
import torch
from torch import tensor, nn, stack, optim, zeros
from torch.nn import functional
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence, pad_packed_sequence
import re
import sys
import csv
import matplotlib.pyplot as plt
import threading
import argparse
from sklearn.metrics import f1_score


def parse_args():
    parser = argparse.ArgumentParser(description='deep learning based sentiment classification')

    controls = parser.add_argument_group()
    controls.add_argument('--name', type=str, default='lstm2lbinodrop', help='name for the model')
    controls.add_argument('--type', type=str, default='lstm', help='network type(mlp, cnn or lstm)')

    hyper_params = parser.add_argument_group()
    hyper_params.add_argument('--lr', type=float, default=0.001, help='learning rate')
    hyper_params.add_argument('--epoch', type=int, default=20, help='training epochs')
    hyper_params.add_argument('--embed-dim', type=int, default=100, help='embedding dimension')
    hyper_params.add_argument('--dropout', type=float, default=0, help='dropout prob')
    hyper_params.add_argument('--clip', type=float, default=5.0, help='gradient clipping')

    mlp_params = hyper_params.add_argument_group()
    mlp_params.add_argument('--hidden-size', type=int, default=256, help='mlp hidden layer size')

    cnn_params = hyper_params.add_argument_group()
    cnn_params.add_argument('--sizes', nargs="+", type=int, default=[2, 3, 4], help="cnn kernel sizes")
    cnn_params.add_argument('--n-kernels', type=int, default=10, help='num each kernel')

    lstm_params = hyper_params.add_argument_group()
    lstm_params.add_argument('--n-layers', type=int, default=2, help='lstm num layers')
    lstm_params.add_argument('--hidden-dim', type=int, default=256, help='lstm hidden state dimension')
    lstm_params.add_argument('--bid', action='store_true', default=True, help='bidirectional lstm')

    args = parser.parse_args()
    return args


class Plotter:
    def __init__(self, save_as):
        self.train_epochs = []
        self.train_losses = []
        self.train_accs = []
        self.valid_epochs = []
        self.valid_losses = []
        self.valid_accs = []
        self.save_as = save_as

    def plot(self):
        plt.clf()
        fig = plt.figure('loss/acc-epoch')
        plt.suptitle('Loss / Accuracy - Epoch Curves')
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        ax1.set_title('Loss - Epoch Curves')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss')
        ax2.set_title('Accuracy - Epoch Curves')
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('accuracy')
        ax2.set_ylim(0, 1)
        ax1.plot(self.train_epochs, self.train_losses, label='train_loss')
        ax1.plot(self.valid_epochs, self.valid_losses, label='valid_loss')
        ax1.legend()
        ax2.plot(self.train_epochs, self.train_accs, label='train_acc')
        ax2.plot(self.valid_epochs, self.valid_accs, label='valid_acc')
        ax2.legend()
        plt.tight_layout()
        plt.savefig(self.save_as)

    def update(self, e, data_type, losses, accs):
        """update data in one epoch"""
        if data_type == 'train':
            for i in range(len(losses)):
                self.train_epochs.append((i+1)/len(losses) + e)
            self.train_losses.extend(losses)
            self.train_accs.extend(accs)
        elif data_type == 'valid':
            for i in range(len(losses)):
                self.valid_epochs.append((i+1)/len(losses) + e)
            self.valid_losses.extend(losses)
            self.valid_accs.extend(accs)


def clean(text):
    punc_re = r'["#$%&()*+,-./:;<=>?@\[\\\]^_`{|}~]'
    digit_re = r'[0-9]'
    text = text.lower()  # lower letters
    text = re.sub(punc_re, ' ', text)  # eliminate punctuations
    text = re.sub('á', '', text)  # eliminate chars making no sense
    text = re.sub(digit_re, '', text)  # eliminate digits
    text = re.sub(r'\s+', ' ', text)  # reduce multiple spaces
    return text


class Vocab:
    def __init__(self, args):
        self.vocab = []
        self.word2idx = {}
        self.labels = []
        self.label2idx = {}

    def __len__(self):
        return len(self.vocab)

    def build(self):
        train_csv_path = './isear_v2/isear_train.csv'
        train_csv = open(train_csv_path, 'r')
        all_words = []
        all_labels = []
        with train_csv:
            reader = csv.DictReader(train_csv)
            for item in reader:
                text = item['sentence']
                # pass empty ones
                if not text:
                    continue
                # pass those strange texts: '[...]'
                if text[0] == '[':
                    continue
                text = clean(text)
                all_words.extend(text.split())
                all_labels.append(item['label'])
        word_counts = Counter(all_words)
        vocab = sorted(word_counts, key=word_counts.get, reverse=True)  # vocabulary from training texts
        vocab.append('<unk>')  # unknown word
        vocab.insert(0, '<pad>')  # padding word
        word2idx = {word: i for i, word in enumerate(vocab)}  # map from word string to vocab index
        label_counts = Counter(all_labels)
        labels = sorted(label_counts, key=label_counts.get, reverse=True)  # label vocab
        labels.append('other')  # other sentiment
        label2idx = {label: i for i, label in enumerate(labels)}  # map from label string to vocab index
        self.vocab = vocab
        self.word2idx = word2idx
        self.labels = labels
        self.label2idx = label2idx

    def encode(self, text):
        text = clean(text)
        word_list = text.split()
        encoded = []
        for word in word_list:
            encoded.append(self.word2idx.get(word, self.word2idx['<unk>']))
        return encoded

    def get_label_index(self, label):
        return self.label2idx.get(label, self.label2idx['other'])


class RNNDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def padding_collate(data):
    data.sort(key=lambda x: len(x[0]), reverse=True)
    seq_lens = [len(x[0]) for x in data]
    padded_seq = pad_sequence([tensor(x[0]) for x in data], batch_first=True, padding_value=0)
    return padded_seq.long(), tensor(seq_lens).long(), tensor([[x[1]] for x in data]).long()
    # both word and label are represented by index


def preprocess_data(data_type, vocab):
    csv_path = ''
    if data_type == 'train':
        csv_path = './isear_v2/isear_train.csv'
    elif data_type == 'valid':
        csv_path = './isear_v2/isear_valid.csv'
    if data_type == 'test':
        csv_path = './isear_v2/isear_test.csv'
    f = open(csv_path, 'r')
    texts = []
    labels = []
    with f:
        reader = csv.DictReader(f)
        for item in reader:
            text = item['sentence']
            label = item['label']
            # pass empty ones
            if not text:
                continue
            # pass those strange texts: '[...]'
            if text[0] == '[':
                continue
            text = vocab.encode(text)
            texts.append(text)
            labels.append(vocab.get_label_index(label))
    return texts, labels    # both list


class LSTM(nn.Module):
    def __init__(self, name, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, bidirectional, dropout_prob):
        super(LSTM, self).__init__()

        self.name = name
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=dropout_prob, batch_first=True, bidirectional=bidirectional)

        if bidirectional:
            self.fc = nn.Linear(hidden_dim * 2, output_size)
        else:
            self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, seqs, lens):
        batch_size = seqs.size(0)  # batch first
        embeds = self.embedding(seqs)
        packed = pack_padded_sequence(input=embeds, lengths=lens, batch_first=True)
        packed_out, _ = self.lstm(packed)
        out, lens = pad_packed_sequence(packed_out, batch_first=True)
        # out: (batch_size * seq_len * hidden_size)
        out = stack([out[i][lens[i] - 1] for i in range(0, batch_size)])    # get state of last time step
        out = self.fc(out)
        out = self.sigmoid(out)
        out = self.softmax(out)
        return out


class CNN(nn.Module):
    def __init__(self, name, vocab_size, output_size, embedding_dim, n_kernels, kernel_sizes, dropout_prob):
        super(CNN, self).__init__()

        self.name = name
        self.output_size = output_size
        self.n_kernels = n_kernels
        self.kernel_sizes = kernel_sizes

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv2d(1, n_kernels, (k, embedding_dim)) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(len(kernel_sizes) * n_kernels, output_size)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, seqs, lens):
        batch_size = seqs.size(0)  # batch first
        embeds = self.embedding(seqs)
        out = embeds.unsqueeze(1)
        out = [functional.relu(conv(out)).squeeze(3) for conv in self.convs]
        out = [functional.max_pool1d(line, line.size(2)).squeeze(2) for line in out]
        out = torch.cat(out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        out = self.sigmoid(out)
        out = self.softmax(out)
        return out


class MLP(nn.Module):
    def __init__(self, name, vocab_size, output_size, embedding_dim, hidden_size):
        super(MLP, self).__init__()

        self.name = name
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, seqs, lens):
        batch_size = seqs.size(0)  # batch first
        embeds = self.embedding(seqs)
        mean = embeds.mean(dim=1)
        out = self.fc1(mean)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = self.softmax(out)
        return out


def calc_accuracy(predicts, labels):  # both represented by indices
    count = 0
    for i in range(len(predicts)):
        if predicts[i] == labels[i]:
            count += 1
    acc = count / len(predicts)
    return acc


def save(net, e=0, optimizer=None):
    save_as = './model/'+net.name+'.tar'
    save_dict = {'model_state_dict': net.state_dict()}
    if e > 0:
        save_dict['epoch'] = e
    if optimizer is not None:
        save_dict['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(save_dict, save_as)
    print('Model has been successfully saved as {}.'.format(save_as))


def train(net, train_loader, criterion, optimizer, epoch, clip, valid_loader=None, plotter=None):
    max_valid_acc = 0.
    for e in range(epoch):
        train_losses = []
        train_accs = []
        valid_losses = []
        valid_accs = []
        print('Now entering training epoch {}...'.format(e))

        for seqs, lens, labels in train_loader:
            net.zero_grad()  # clear accumulated gradients
            out = net(seqs, lens)   # forward

            # transform labels into one hot
            batch_size = out.size(0)
            num_classes = out.size(1)
            one_hot = zeros(batch_size, num_classes).scatter_(1, labels, 1)

            # calculate loss and back propagation
            loss = criterion(out, one_hot)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), clip)   # perform gradient clipping
            optimizer.step()

            train_losses.append(loss.item())
            train_accs.append(calc_accuracy(out.max(axis=1)[1], labels))

        print('Epoch {} finished. On this epoch:'.format(e))
        print('Training loss: {:.6f}'.format(np.mean(train_losses)))
        print('Training accuracy: {:.6f}'.format(np.mean(train_accs)))

        if valid_loader is not None:
            # validate
            print('Now begin validation for epoch {}...'.format(e+1))
            net.eval()
            for valid_seqs, valid_lens, valid_labels in valid_loader:
                valid_out = net(valid_seqs, valid_lens)
                # transform labels into one hot
                valid_batch_size = valid_out.size(0)
                num_classes = valid_out.size(1)
                valid_one_hot = zeros(valid_batch_size, num_classes).scatter_(1, valid_labels, 1)
                valid_loss = criterion(valid_out, valid_one_hot)
                valid_losses.append(valid_loss.item())
                valid_acc = calc_accuracy(valid_out.max(axis=1)[1], valid_labels)
                valid_accs.append(valid_acc)

            print('Finished validation for epoch {}, stats:'.format(e))
            print('Validation loss: {:.6f}'.format(np.mean(valid_losses)))
            print('Validation accuracy: {:.6f}'.format(np.mean(valid_accs)))
            # save checkpoint if performance on valid set is the best by now
            if np.mean(valid_accs) > max_valid_acc:
                print('Best performance detected, saving model...')
                max_valid_acc = np.mean(valid_accs)
                save(net, e+1, optimizer)

            net.train()

        if plotter is not None:
            plotter.update(e, 'train', train_losses, train_accs)
            if valid_loader is not None:
                plotter.update(e, 'valid', valid_losses, valid_accs)
            plotter.plot()
            print('Curve image after epoch {} has been saved.'.format(e+1))


def test(net, test_loader):
    label_list = []
    pred_list = []
    with torch.no_grad():
        for seqs, lens, labels in test_loader:
            out = net(seqs, lens)  # forward
            pred_list.extend([pred.item() for pred in out.max(axis=1)[1]])
            label_list.extend([label.item() for label in labels])
    return calc_accuracy(pred_list, label_list), f1_score(label_list, pred_list, average='macro'), f1_score(label_list, pred_list, average='micro')


def main():
    args = parse_args()

    # build vocabulary
    vocab = Vocab(args)
    vocab.build()

    # preprocess data, build dataset and loader
    train_texts, train_labels = preprocess_data('train', vocab)
    valid_texts, valid_labels = preprocess_data('valid', vocab)
    test_texts, test_labels = preprocess_data('test', vocab)
    train_set = RNNDataset(train_texts, train_labels)
    valid_set = RNNDataset(valid_texts, valid_labels)
    test_set = RNNDataset(test_texts, test_labels)
    print('Training set has been loaded with', len(train_set), 'samples.')
    print('Validation set has been loaded with', len(valid_set), 'samples.')
    print('Test set has been loaded with', len(test_set), 'samples.')
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, collate_fn=padding_collate)
    valid_loader = DataLoader(valid_set, batch_size=16, shuffle=True, collate_fn=padding_collate)
    test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=True, collate_fn=padding_collate)

    # init net
    if args.type == 'lstm':
        net = LSTM(args.name, len(vocab), len(vocab.labels),
                   args.embed_dim, args.hidden_dim, args.n_layers, args.bid, args.dropout)
    elif args.type == 'cnn':
        net = CNN(args.name, len(vocab), len(vocab.labels), args.embed_dim, args.n_kernels, args.sizes, args.dropout)
    else:
        net = MLP(args.name, len(vocab), len(vocab.labels), args.embed_dim, args.hidden_size)
    print('Neural network created, details:', net)

    # set criterion and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    print('Criterion and optimizer set.')

    # init plotter
    curve_save_path = './curve/'+args.name+'.jpg'
    plotter = Plotter(curve_save_path)
    print('Plotter created, curve images will be saved as {}.'.format(curve_save_path))

    # train
    net.train()
    print('Training started.')
    train(net, train_loader, criterion, optimizer, args.epoch, args.clip, valid_loader, plotter)

    # test
    net.eval()
    print('Running on test set...')
    acc, f1_macro, f1_micro = test(net, test_loader)
    print('Test finished, stats:')
    print('Total accuracy: {:.6f}'.format(acc))
    print('F1 score (macro): {:.6f}'.format(f1_macro))
    print('F1 score (micro): {:.6f}'.format(f1_micro))


if __name__ == "__main__":
    main()
