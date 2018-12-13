import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import time
from torch.autograd import Variable
import sys

import math

OutputFile= "output_log.txt"
OF = open(OutputFile, 'w')

def printing(text):
    print(text)
    OF.write('%s\n' %text)

def showPlot(points, name):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig('%s.pdf' % name)
    plt.close()

def showAttention(input_word, output_word, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_word.split(' ') + ['<STOP>'], rotation=90)
    ax.set_yticklabels([''] + output_word)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

def evaluateAndShowAttention(device, model_name, encoder, decoder, Vocab_source, Vocab_target, input_word):
    output_word, attentions = evaluateWord(device, model_name, encoder, decoder, Vocab_source, Vocab_target, input_word)
    print('input =', input_word)
    print('output =', ''.join(output_word))
    showAttention(input_word, output_word, attentions)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

IND_START=0
IND_STOP=1
IND_UNK=2
MAX_LENGTH=20

class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.char2index = {}
        self.char2count = {}
        self.index2char = {0: "START", 1: "STOP", 2: 'UNK'}
        self.vocab_size = 3
        self.max_word_size = 5

    def addWord(self, word):
        for char in word:
            self.addChar(char)
        if len(word) > self.max_word_size:
            self.max_word_size = len(word)

    def addChar(self, char):
        if char not in self.char2index:
            self.char2index[char] = self.vocab_size
            self.char2count[char] = 1
            self.index2char[self.vocab_size] = char
            self.vocab_size += 1
        else:
            self.char2count[char] += 1

def indexesFromWord(Vocab, word):
    return [IND_UNK if char not in Vocab.char2index else Vocab.char2index[char] for char in word]

def tensorFromWord(Vocab, word, device):
    indexes = indexesFromWord(Vocab, word)
    indexes.append(IND_STOP)
    return torch.tensor(indexes, dtype=torch.long).view(-1, 1).to(device)

def read_data(filepath, reverse):
    try:
        f = open(filepath)
    except IOError:
        print("File not found or path is incorrect")

    max_len_s = 0
    max_len_t = 0
    vocab_s=Vocabulary('arabic')
    vocab_t=Vocabulary('english')

    source_words=[]
    target_words=[]
    for line in f:
        word_pair = line.split()

        if (reverse == True):
            source_words.append(word_pair[0][::-1])
        else:
            source_words.append(word_pair[0])
        target_words.append(word_pair[1])

        vocab_s.addWord(word_pair[0])
        vocab_t.addWord(word_pair[1])
    f.close()

    return  source_words, target_words, vocab_s, vocab_t

class EncoderLSTM(torch.nn.Module):
    def __init__(self, vocab_s_size, embed_size, hidden_size, device):
        super(EncoderLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.device = device
        self.hidden  = self.initHidden()

        self.embed = torch.nn.Embedding(vocab_s_size, embed_size)
        self.lstm = torch.nn.LSTM(embed_size, hidden_size, 1)

    def forward(self, input):
        embeddings = self.embed(input).view(1,1,-1)
        out, self.hidden = self.lstm(embeddings, self.hidden)
        return out

    def initHidden(self):
        h0 = torch.zeros(1, 1, self.hidden_size, device=self.device)
        c0 = torch.zeros(1, 1, self.hidden_size, device=self.device)
        return (h0, c0)

class DecoderLSTM(torch.nn.Module):
    def __init__(self, vocab_t_size, embed_size, hidden_size, device):
        super(DecoderLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.device = device
        self.hidden = self.initHidden()

        self.embed = torch.nn.Embedding(vocab_t_size, embed_size)
        self.lstm = torch.nn.LSTM(embed_size, hidden_size, 1)
        self.fc = torch.nn.Linear(hidden_size, vocab_t_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input):
        embeddings = self.embed(input).view(1,1,-1)
        out, self.hidden = self.lstm(embeddings, self.hidden)
        out = self.fc(out)
        out = self.softmax(out[0])
        return out

    def initHidden(self):
        h0 = torch.zeros(1, 1, self.hidden_size, device=self.device)
        c0 = torch.zeros(1, 1, self.hidden_size, device=self.device)
        return (h0, c0)

class EncoderBILSTM(torch.nn.Module):
    def __init__(self, vocab_s_size, embed_size, hidden_size, device, num_layers, dropout):
        super(EncoderBILSTM, self).__init__()

        self.hidden_size = hidden_size
        self.device = device
        self.num_layers = num_layers
        self.hidden = self.initHidden()

        self.embed = torch.nn.Embedding(vocab_s_size, embed_size)
        self.lstm = torch.nn.LSTM(embed_size, hidden_size, num_layers, bidirectional=True, dropout=dropout)

    def forward(self, input):
        embeddings = self.embed(input).view(1,1,-1)
        out, self.hidden = self.lstm(embeddings, self.hidden)
        return out

    def initHidden(self):
        h0 = torch.zeros(self.num_layers*2, 1, self.hidden_size, device=self.device)
        c0 = torch.zeros(self.num_layers*2, 1, self.hidden_size, device=self.device)
        return (h0, c0)

class DecoderBILSTM(torch.nn.Module):
    def __init__(self, vocab_t_size, embed_size, hidden_size, device, num_layers, dropout):
        super(DecoderBILSTM, self).__init__()

        self.hidden_size = hidden_size
        self.device = device
        self.num_layers = num_layers
        self.hidden  = self.initHidden()

        self.embed = torch.nn.Embedding(vocab_t_size, embed_size)
        self.lstm = torch.nn.LSTM(embed_size, hidden_size, num_layers, bidirectional=True, dropout=dropout)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden_size*2, vocab_t_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input):
        embeddings = self.embed(input).view(1,1,-1)
        out, self.hidden = self.lstm(embeddings, self.hidden)
        out = self.dropout(out)
        out = self.fc(out)
        out = self.softmax(out[0])
        return out

    def initHidden(self):
        h0 = torch.zeros(self.num_layers*2, 1, self.hidden_size, device=self.device)
        c0 = torch.zeros(self.num_layers*2, 1, self.hidden_size, device=self.device)
        return (h0, c0)

class AttnDecoderBILSTM(torch.nn.Module):
    def __init__(self, vocab_t_size, embed_size, hidden_size, device, num_layers, dropout, max_length):
        super(AttnDecoderBILSTM, self).__init__()

        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.device = device
        self.num_layers = num_layers
        self.hidden = self.initHidden()
        self.max_length = max_length


        self.embed = torch.nn.Embedding(vocab_t_size, embed_size)
        self.dropout = torch.nn.Dropout(dropout)

        self.lstm = torch.nn.LSTM(embed_size, hidden_size, num_layers, bidirectional=True, dropout=dropout)

        self.attn = torch.nn.Linear(hidden_size, hidden_size*2)
        self.Whc = torch.nn.Linear(hidden_size * 3, hidden_size)
        self.Ws = torch.nn.Linear(hidden_size, vocab_t_size)

    def forward(self, input, encoder_outputs):
        embeddings = self.embed(input).view(1,1,-1)
        embeddings = self.dropout(embeddings)

        out, (h, c) = self.lstm(embeddings, self.hidden)

        self.hidden = (h,c)

        attn_prod = torch.mm(self.attn(h)[0], encoder_outputs.t())
        attn_weights = torch.nn.functional.softmax(attn_prod, dim=1)
        context = torch.mm(attn_weights, encoder_outputs)

        hc = torch.cat([h[0], context], dim=1)
        aux = self.Whc(hc)
        out_hc = torch.tanh(aux)
        output = torch.nn.functional.log_softmax(self.Ws(out_hc), dim=1)

        return output, attn_weights

    def initHidden(self):
        h0 = torch.zeros(self.num_layers*2, 1, self.hidden_size, device=self.device)
        c0 = torch.zeros(self.num_layers*2, 1, self.hidden_size, device=self.device)
        return (h0, c0)

def evaluateWord(device, model_name, encoder, decoder, Vocab_source, Vocab_target, word):
    with torch.no_grad():
        input_tensor = tensorFromWord(Vocab_source, word, device)

        input_length = input_tensor.size()[0]
        encoder.hidden = encoder.initHidden()

        max_length = Vocab_target.max_word_size
        if model_name == 'LSTM':
            encoder_outputs = torch.zeros(max_length+2, encoder.hidden_size).to(device)
        else:
            encoder_outputs = torch.zeros(max_length+2, encoder.hidden_size*2).to(device)

        for ei in range(input_length):
            encoder_output = encoder(input_tensor[ei])
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[IND_START]]).to(device)  # SOS

        decoder.hidden = encoder.hidden

        decoded_word = []
        decoder_attentions = torch.zeros(max_length+2, max_length+2).to(device)

        for di in range(max_length+2):
            if model_name == 'AttnBILSTM':
                decoder_output, decoder_attention = decoder(decoder_input, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
            else:
                decoder_output = decoder(decoder_input)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == IND_STOP:
                decoded_word.append('<STOP>')
                break
            elif topi.item() > Vocab_target.vocab_size:
                print('Unknown char found: %s' %(topi.item()))
                decoded_word.append('<UNK>')
            else:
                decoded_word.append(Vocab_target.index2char[topi.item()])
            decoder_input = topi.squeeze().detach()

        return decoded_word, decoder_attentions[:di+1]

def evaluate(device, model_name, encoder, decoder, Vocab_source, Vocab_target, source_words, target_words):
    correct_words=0
    total_words = len(source_words)
    for i_word in range(total_words):
        output, attentions = evaluateWord(device, model_name, encoder, decoder, Vocab_source, Vocab_target, source_words[i_word])
        output_word = ''.join(output[:len(output)-1])
        if (output_word == target_words[i_word]):
            correct_words += 1

    accuracy = correct_words * 1.0 / total_words *100

    return correct_words, accuracy

def trainWord(device, model_name, input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    if model_name == 'LSTM':
        encoder_outputs = torch.zeros(max_length+2, encoder.hidden_size).to(device)
    else:
        encoder_outputs = torch.zeros(max_length+2, encoder.hidden_size*2).to(device)

    loss = 0

    encoder.hidden = encoder.initHidden()

    for ei in range(input_length):
        encoder_output = encoder(input_tensor[ei])
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder.hidden = encoder.hidden

    decoder_input = torch.tensor([[IND_START]], device=device)

    # Teacher forcing: Feed the target as the next input
    for di in range(target_length):
        if model_name == 'AttnBILSTM':
            decoder_output, decoder_attention = decoder(decoder_input, encoder_outputs)
        else:
            decoder_output = decoder(decoder_input)
        loss += criterion(decoder_output, target_tensor[di])
        decoder_input = target_tensor[di]  # Teacher forcing
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def trainModel(device, Vocab_source, Vocab_target, source_words_train, target_words_train, source_w_eval, target_w_eval, model_name, arg_optimizer, learning_rate, dropout_prob, hidden_size, embed_size, num_epochs, reverse):
    print_every=1000
    plot_every=100

    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    trainTime = time.time()

    if model_name == 'LSTM':
        printing('Training %s model with %s source words (eta: %f, num_epochs: %d, optimizer: %s)...' % (model_name, 'reversed' if reverse==True else 'regular', learning_rate, num_epochs, arg_optimizer))
        encoder = EncoderLSTM(Vocab_source.vocab_size, embed_size, hidden_size, device).to(device)
        decoder = DecoderLSTM(Vocab_target.vocab_size, embed_size, hidden_size, device).to(device)
    elif model_name == 'BILSTM':
        num_layers = 2
        printing('Training %s model with %s source words (num_layers: %d, eta: %f, num_epochs: %d, dropout_prob: %.2f, optimizer: %s)...' % (model_name, 'reversed' if reverse==True else 'regular', num_layers, learning_rate, num_epochs, dropout_prob, arg_optimizer))
        encoder = EncoderBILSTM(Vocab_source.vocab_size, embed_size, hidden_size, device, num_layers, dropout_prob).to(device)
        decoder = DecoderBILSTM(Vocab_target.vocab_size, embed_size, hidden_size, device, num_layers, dropout_prob).to(device)
    elif model_name == 'AttnBILSTM':
        num_layers = 2
        printing('Training %s model with %s source words (num_layers: %d, eta: %f, num_epochs: %d, dropout_prob: %.2f, optimizer: %s)...' % (model_name, 'reversed' if reverse==True else 'regular', num_layers, learning_rate, num_epochs, dropout_prob, arg_optimizer))
        encoder = EncoderBILSTM(Vocab_source.vocab_size, embed_size, hidden_size, device, num_layers, dropout_prob).to(device)
        decoder = AttnDecoderBILSTM(Vocab_target.vocab_size, embed_size, hidden_size, device, num_layers, dropout_prob, MAX_LENGTH).to(device)
    else:
        printing('Wrong Model Name')
        sys.exit()

    if arg_optimizer == 'SGD':
        encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=learning_rate)
    elif arg_optimizer == 'Adam':
        encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    elif arg_optimizer == 'Adagrad':
        encoder_optimizer = torch.optim.Adagrad(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = torch.optim.Adagrad(decoder.parameters(), lr=learning_rate)
    else:
        printing('Wrong Optimizer')
        sys.exit()

    criterion = torch.nn.NLLLoss()

    n_iters = len(source_words_train)

    random_ordering = list(range(n_iters))
    np.random.shuffle(random_ordering)

    acc_per_epoch = []
    loss_per_epoch = []
    for epoch in range(num_epochs):
        printing('Beginning epoch %d' %(epoch+1))
        start = time.time()
        aux_iter=1
        for iter in random_ordering:
            input_tensor =  tensorFromWord(Vocab_source, source_words_train[iter], device)
            target_tensor =  tensorFromWord(Vocab_target, target_words_train[iter], device)

            loss = trainWord(device, model_name, input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, Vocab_target.max_word_size)
            print_loss_total += loss
            plot_loss_total += loss

            if aux_iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                printing('\t%s (%d %d%%) %.4f' % (timeSince(start, aux_iter / n_iters), aux_iter, aux_iter / n_iters * 100, print_loss_avg))
            aux_iter += 1

        correct_words_eval, accuracy_eval = evaluate(device, model_name, encoder, decoder, Vocab_source, Vocab_target, source_w_eval, target_w_eval)
        printing('Epoch %d: %d (%.2f%%) correctly predicted words' %(epoch+1, correct_words_eval, accuracy_eval))

        acc_per_epoch.append(accuracy_eval)
        if accuracy_eval < 2:
            printing('Accuracy is not high enough to warrant continuing')
            break

    endTime = time.time()
    printing('Training Time: {0:0.2f} seconds'.format(endTime-trainTime))

    return acc_per_epoch, (endTime-trainTime), encoder, decoder

def main():
    """Main function."""
    import argparse
    import sys

    # Parse arguments.
    parser = argparse.ArgumentParser()

    parser.add_argument('device', type=str)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--tune', action='store_const', const=True, default=False)
    parser.add_argument('--reverse', action='store_const', const=True, default=False)
    parser.add_argument('--read_vocab', action='store_const', const=True, default=False)
    parser.add_argument('--dropout_prob', type=float, default=0.0)
    parser.add_argument('--optimizer', type=str, default="SGD")
    parser.add_argument('--datapath', type=str, default="datasets")
    parser.add_argument('--model', type=str, default="LSTM")
    parser.add_argument('--learning_rate', type=float, default=0.01)

    args = vars(parser.parse_args())
    printing(args)

    device_arg = args['device']
    device_id = args['device_id']
    model_name = args['model']
    arg_optimizer = args['optimizer']
    dropout_prob = args['dropout_prob']
    tune = args['tune']
    reverse = args['reverse']
    read_vocab = args['read_vocab']
    learning_rate = args['learning_rate']
    num_epochs = args['num_epochs']
    datapath = args['datapath']

    #check if selected device is GPU and if CUDA is available
    if (device_arg == 'gpu'):
        if torch.cuda.is_available():
            device = torch.device("cuda:%d" %(device_id))
            printing(torch.cuda.get_device_name(device_id))
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")


    printing(device)

    np.random.seed(42)

    startTime = time.time()

    printing('Loading data...')

    source_words_train, target_words_train, vocab_s_train, vocab_t_train = read_data('%s/ar2en-train.txt' %(datapath), reverse)
    source_eval, target_eval, vocab_s_eval, vocab_t_eval = read_data('%s/ar2en-eval.txt' %(datapath), reverse)
    source_test, target_test, vocab_s_test, vocab_t_test = read_data('%s/ar2en-test.txt' %(datapath), reverse)
    finishLoadTime = time.time()
    printing('Loading Data Time: {0:0.2f} seconds'.format(finishLoadTime-startTime))

    if read_vocab == True:
        printing('Source vocabulary:')
        printing(vocab_s_train.index2char)
        # for vocab in vocab_s_train.index2char:
            # print(vocab_s_train.index2char[vocab])
        printing('Size: %d' %(vocab_s_train.vocab_size))

        printing('\nTarget vocabulary:')
        printing(vocab_t_train.index2char)
        printing('Size: %d' %(vocab_t_train.vocab_size))
        sys.exit()
    else:
        hidden_size = 50
        embed_size = 30
        if (tune == True):
            dropout_probs = [0, 0.25, 0.5]
            learning_rates = [0.001, 0.005, 0.01, 0.1]
            optimizers = ['SGD', 'Adam']

            num_epochs_tune = 10

            best_acc_per_optimizer = [0, 0, 0]
            best_epoch_per_optimizer =  [-1, -1, -1]
            best_learning_rate_per_optimizer = [-1.0, -1.0, -1.0]
            best_dropout_prob_per_optimizer = [-1.0, -1.0, -1.0]
            best_time_per_optimizer = [-1.0, -1.0, -1.0]
            printing('Starting Hyperparameter Tuning for %s model' %(model_name))

            for opt_id, arg_optimizer in enumerate(optimizers):
                learning_rate = learning_rates[0]
                dropout_prob = dropout_probs[0]

                for learning_rate in learning_rates:
                    acc_per_epoch, trainingTime, encoder, decoder = trainModel(device, vocab_s_train, vocab_t_train, source_words_train, target_words_train, source_eval, target_eval, model_name, arg_optimizer, learning_rate, dropout_prob, hidden_size, embed_size, num_epochs_tune, reverse)

                    best_accuracy_test = max(acc_per_epoch)
                    if best_accuracy_test > best_acc_per_optimizer[opt_id]:
                        best_acc_per_optimizer[opt_id] = best_accuracy_test
                        best_epoch_per_optimizer[opt_id] =  acc_per_epoch.index(best_accuracy_test)
                        best_learning_rate_per_optimizer[opt_id] = learning_rate
                        best_dropout_prob_per_optimizer[opt_id] = dropout_prob
                        best_time_per_optimizer[opt_id] = trainingTime

                learning_rate = best_learning_rate_per_optimizer[opt_id]

                if not model_name == 'LSTM':
                    for dropout_prob in dropout_probs[1:]:
                        acc_per_epoch, trainingTime, encoder, decoder = trainModel(device, vocab_s_train, vocab_t_train, source_words_train, target_words_train, source_eval, target_eval, model_name, arg_optimizer, learning_rate, dropout_prob, hidden_size, embed_size, num_epochs_tune, reverse)

                        best_accuracy_test = max(acc_per_epoch)
                        if best_accuracy_test > best_acc_per_optimizer[opt_id]:
                            best_acc_per_optimizer[opt_id] = best_accuracy_test
                            best_epoch_per_optimizer[opt_id] =  acc_per_epoch.index(best_accuracy_test)
                            best_dropout_prob_per_optimizer[opt_id] = dropout_prob
                            best_time_per_optimizer[opt_id] = trainingTime

                printing('Best results for %s optimizer (%s): (eta: %f, dropout_prob: %f, epoch_number: %d, accuracy: %.4f, time: %.2f s)' %(arg_optimizer, model_name, best_learning_rate_per_optimizer[opt_id], best_dropout_prob_per_optimizer[opt_id], best_epoch_per_optimizer[opt_id], best_acc_per_optimizer[opt_id], best_time_per_optimizer[opt_id]))

            finishTuneTime = time.time()
            printing('\nTuning the hyperparameters has ended (duration:  {0:0.2f} seconds)'.format(finishTuneTime-finishLoadTime))
            best_acc = max(best_acc_per_optimizer)
            opt_id_aux = best_acc_per_optimizer.index(best_acc)
            best_optimizer = optimizers[opt_id_aux]
            best_learning_rate = best_learning_rate_per_optimizer[opt_id_aux]
            best_dropout_prob = best_dropout_prob_per_optimizer[opt_id_aux]
            best_epoch = best_epoch_per_optimizer[opt_id_aux]
            best_time = best_time_per_optimizer[opt_id_aux]

            printing('Best results %s: (optimizer: %s, eta: %f, dropout_prob: %f, epoch_number: %d, accuracy: %.4f, time: %.2f s)' %(model_name, best_optimizer, best_learning_rate, best_dropout_prob, best_epoch, best_acc, best_time))

            learning_rate = best_learning_rate
            num_epochs = best_epoch
            arg_optimizer = best_optimizer
            dropout_prob = best_dropout_prob

        acc_per_epoch, trainingTime, encoder, decoder = trainModel(device, vocab_s_train, vocab_t_train, source_words_train, target_words_train, source_eval, target_eval, model_name, arg_optimizer, learning_rate, dropout_prob, hidden_size, embed_size, num_epochs, reverse)

        showPlot(acc_per_epoch, 'accuracy_%s_%s_%s_%f_%f.pdf' %(model_name, 'reversed' if reverse == True else 'regular', arg_optimizer, learning_rate, dropout_prob))

        correct_words_test, accuracy_test = evaluate(device, model_name, encoder, decoder, vocab_s_train, vocab_t_train, source_test, target_test)
        printing('Final Test Word Accuracy: %d (%.2f%%) correctly predicted words' %(correct_words_test, accuracy_test))

    OF.close()

if __name__ == "__main__":
    main()
