import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from torch.autograd import Variable
import sys

OutputFile= "output_log.txt"
OF = open(OutputFile, 'w')

def printing(text):
    print(text)
    OF.write('%s\n' %text)

def read_data(filepath, partitions=None, pairwise_features=False):
    """Read the OCR dataset."""
    labels = {}
    f = open(filepath)
    X = []
    y = []
    for line in f:
        line = line.rstrip('\t\n')
        fields = line.split('\t')
        letter = fields[1]
        if letter in labels:
            k = labels[letter]
        else:
            k = len(labels)
            labels[letter] = k
        partition = int(fields[5])
        if partitions is not None and partition not in partitions:
            continue
        x = np.empty((1,16,8))
        aux_x = np.array([float(v) for v in fields[6:]])
        aux_x2 = np.reshape(aux_x,(16,8))
        x[0,:,:] = aux_x2

        X.append(x)
        y.append(k)
    f.close()
    l = ['' for k in labels]
    for letter in labels:
        l[labels[letter]] = letter
    return X, y, l

class CNN_BILSTMNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, num_layers, num_classes, dropout):
        super(CNN_BILSTMNet, self).__init__()
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = num_classes

        self.layer_cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.dropout_cnn = torch.nn.Dropout(dropout)
        self.fc_cnn = torch.nn.Linear(20*8*4, hidden_size_1)

        self.lstm = torch.nn.LSTM(hidden_size_1, hidden_size_2, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden_size_2*2, num_classes)

    def forward(self, x, device):
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size_2).to(device) # 2 for bidirection
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size_2).to(device)

        out = self.layer_cnn(x)
        out = out.reshape(out.size(0), -1)
        out = self.dropout_cnn(out)
        out = self.fc_cnn(out)
        out = out.reshape(out.size(0), 1, -1)

        out, _ = self.lstm(out, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        out = self.dropout(out[:, -1, :])  # Decode the hidden state of the last time step
        out = torch.nn.functional.log_softmax(self.fc(out), dim=1)
        # out = self.sftm(out)
        return out

class BILSTMNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, num_layers, num_classes, dropout):
        super(BILSTMNet, self).__init__()
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.num_layers = num_layers

        self.input_layer = torch.nn.Linear(input_size, hidden_size_1)
        self.lstm = torch.nn.LSTM(hidden_size_1, hidden_size_2, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden_size_2*2, num_classes)
        # self.sftm = torch.nn.Softmax(dim=1)
    def forward(self, x, device):
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size_2).to(device) # 2 for bidirection
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size_2).to(device)

        out = torch.nn.functional.relu(self.input_layer(x))
        out, _ = self.lstm(out, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        out = self.dropout(out[:, -1, :])  # Decode the hidden state of the last time step
        out = self.fc(out)
        out = torch.nn.functional.log_softmax(out, dim=1)
        # out = self.sftm(out)
        return out

def plotFiguresAccLoss(loss_per_epoch, acc_per_epoch, num_epochs, model_name, device, learning_rate, batch_size, optimizer):
    printing('FigureACCLossFunc: model name: %s, optimizer: %s' %(model_name, optimizer))
    plt.figure(1)
    plt.plot(range(1, num_epochs+1), loss_per_epoch, 'bo-')
    plt.title('Dev Losses')
    name = 'loss_%s_%s_%s_%s_%s' % (device, model_name, learning_rate, batch_size, optimizer)

    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig('%s.pdf' % name)
    plt.close()

    plt.figure(2)
    plt.plot(range(1, num_epochs+1), acc_per_epoch, 'bo-')
    plt.title('Dev Accuracy')
    name = 'acc_%s_%s_%s_%s_%s' % (device, model_name, learning_rate, batch_size, optimizer)

    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.savefig('%s.pdf' % name)
    plt.close()

def evaluate_pytorch(device, model_name, model, X, y, sequence_length, input_size):
    """Evaluate model on data."""

    test_x = torch.tensor(X, dtype=torch.float).to(device)
    if (model_name == 'BILSTM'):
        test_x = test_x.reshape(-1, sequence_length, input_size)

    test_y = torch.tensor(y, dtype=torch.long).to(device)

    loss_function = torch.nn.CrossEntropyLoss()

    y_pred = model(test_x, device)

    loss = loss_function(y_pred, test_y)
    loss_value = loss.item()
    avg_loss = loss_value / len(y)

    hits = torch.sum(y_pred.argmax(dim=1) == test_y).item()
    accuracy = float(hits) / len(y)

    return accuracy, loss_value

def train_pytorch_model(device, model_name, X_train, y_train, X_dev, y_dev, learning_rate, reg, num_epochs, batch_size, dropout, optimizer_name, sequence_length, input_size, hidden_size_1, hidden_size_2, num_layers):
    num_classes = 26

    if (model_name == 'BILSTM'):
        printing('Creating a BILSTM  Network')
    elif (model_name == 'CNN_BILSTM'):
        printing('Creating a CNN and BILSTM Network')
    else:
        printing('Wrong model name')
        sys.exit()

    trainTime = time.time()
    printing('Training %s model(eta: %f, reg: %f, num_epochs: %d, batch_size: %d, dropout: %f,  optimizer: %s)...' % (model_name, learning_rate, reg, num_epochs, batch_size, dropout, optimizer_name))

    if (device == 'cpu'):
        torch.manual_seed(42)
    else:
        torch.cuda.manual_seed(42)


    if (model_name == 'BILSTM'):
        model = BILSTMNet(input_size, hidden_size_1, hidden_size_2, num_layers, num_classes, dropout).to(device)
    else:
        model = CNN_BILSTMNet(input_size, hidden_size_1, hidden_size_2, num_layers, num_classes, dropout).to(device)

    loss_function = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')

    optimizer_SGD = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=reg)
    optimizer_Adam = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg)
    optimizer_Adagrad= torch.optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=reg)

    if (optimizer_name == 'SGD'):
        optimizer = optimizer_SGD
    elif (optimizer_name == 'Adam'):
        optimizer = optimizer_Adam
    else:
        optimizer = optimizer_Adagrad

    losses = []
    loss_per_epoch = []
    acc_per_epoch = []
    for epoch in range(num_epochs):
        printing('Starting epoch %d' % epoch)
        batch_index = 0
        total_loss = 0
        hits = 0
        running_loss = 0.0
        while batch_index < len(X_train):
            # get the data for this batch
            next_index = batch_index + batch_size
            batch_x = torch.tensor(X_train[batch_index:next_index], dtype=torch.float).to(device)

            if (model_name == 'BILSTM'):
                batch_x = batch_x.reshape(-1, sequence_length, input_size)

            if (batch_index == 0):
                print(batch_x.size())

            batch_labels = torch.tensor(y_train[batch_index:next_index], dtype=torch.long).to(device)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            logits = model(batch_x, device)

            loss = loss_function(logits, batch_labels)
            loss_value = loss.item()
            total_loss += loss_value
            losses.append(loss_value)

            batch_index = next_index

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if batch_index % 999*batch_size == 0:    # print every 2000 mini-batches
                printing('[%d, %5d] loss: %.3f' % (epoch + 1, batch_index + 1, running_loss / 1000))
                running_loss = 0.0

        #evaluate on evaluation dataset
        (acc, avg_loss) = evaluate_pytorch(device, model_name, model, X_dev, y_dev, sequence_length, input_size)

        printing('Epoch loss: %.4f' % avg_loss)
        loss_per_epoch.append(avg_loss)
        printing('Epoch accuracy: %.4f' % acc)
        acc_per_epoch.append(acc)

    endTime = time.time()
    printing('Training Time: {0:0.2f} seconds'.format(endTime-trainTime))

    return loss_per_epoch, acc_per_epoch, (endTime-trainTime), model

def main():
    """Main function."""
    import argparse
    import sys

    # Parse arguments.
    parser = argparse.ArgumentParser()

    parser.add_argument('device', type=str)
    parser.add_argument('--model', type=str, default="BILSTM")
    parser.add_argument('--optimizer', type=str, default="Adam")
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--reg_constant', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--dropout_prob', type=float, default=0.0)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--tune', action='store_const', const=True, default=False)

    args = vars(parser.parse_args())
    printing(args)

    device_arg = args['device']
    num_epochs = args['num_epochs']
    device_id = args['device_id']
    model_name = args['model']
    arg_optimizer = args['optimizer']
    learning_rate = args['learning_rate']
    reg_constant = args['reg_constant']
    batch_size = args['batch_size']
    dropout_prob = args['dropout_prob']
    tune = args['tune']

    pairwise=False

    filepath = '../../letter.data'

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
    X_train, y_train, labels = read_data(filepath, partitions=set(range(8)), pairwise_features=pairwise)

    X_dev, y_dev, _ = read_data(filepath, partitions={8}, pairwise_features=pairwise)
    X_test, y_test, _ = read_data(filepath, partitions={9}, pairwise_features=pairwise)
    finishLoadTime = time.time()

    printing('Loading Data Time: {0:0.2f} seconds'.format(finishLoadTime-startTime))

    # Hyper-parameters
    sequence_length = 16
    input_size = 8
    hidden_size_1 = 64
    hidden_size_2 = 128
    num_layers = 2
    num_classes = 26

    if (tune == True):
        learning_rates = [0.001, 0.005, 0.01, 0.1]
        batch_sizes = [4, 16, 128]
        regs = [0, 0.1, 0.01, 0.001]
        dropout_probs = [0, 0.25, 0.5]
        optimizers = ['SGD', 'Adam', 'Adagrad']

        num_epochs_tune = 20

        printing('Starting Hyperparameter Tuning for %s model' %(model_name))

        best_acc_per_optimizer = [0, 0, 0]
        best_loss_per_optimizer = [-1.0, -1.0, -1.0]
        best_epoch_per_optimizer =  [-1, -1, -1]
        best_learning_rate_per_optimizer = [-1.0, -1.0, -1.0]
        best_reg_per_optimizer = [-1.0, -1.0, -1.0]
        best_batch_size_per_optimizer = [-1, -1, -1]
        best_dropout_prob_per_optimizer = [-1.0, -1.0, -1.0]
        best_optimizer_per_optimizer = ['', '', '']
        best_time_per_optimizer = [-1.0, -1.0, -1.0]

        for opt_id, arg_optimizer in enumerate(optimizers):
            learning_rate = learning_rates[0]
            batch_size = batch_sizes[0]
            dropout_prob = dropout_probs[0]

            for reg_constant in regs:
                (loss_per_epoch, acc_per_epoch, trainingTime, trainedModel) = train_pytorch_model(device, model_name, X_train, y_train, X_dev, y_dev, learning_rate, reg_constant, num_epochs_tune, batch_size, dropout_prob, arg_optimizer, sequence_length, input_size, hidden_size_1, hidden_size_2, num_layers)

                best_accuracy_test = max(acc_per_epoch)
                if best_accuracy_test > best_acc_per_optimizer[opt_id]:
                    best_acc_per_optimizer[opt_id] = best_accuracy_test
                    best_epoch_per_optimizer[opt_id] =  acc_per_epoch.index(best_accuracy_test)
                    best_loss_per_optimizer[opt_id] = loss_per_epoch[best_epoch_per_optimizer[opt_id]]
                    best_learning_rate_per_optimizer[opt_id] = learning_rate
                    best_reg_per_optimizer[opt_id] = reg_constant
                    best_batch_size_per_optimizer[opt_id] = batch_size
                    best_dropout_prob_per_optimizer[opt_id] = dropout_prob
                    best_time_per_optimizer[opt_id] = trainingTime

            reg_constant = best_reg_per_optimizer[opt_id]

            for learning_rate in learning_rates[1:]:
                (loss_per_epoch, acc_per_epoch, trainingTime, trainedModel) = train_pytorch_model(device, model_name, X_train, y_train, X_dev, y_dev, learning_rate, reg_constant, num_epochs_tune, batch_size, dropout_prob, arg_optimizer, sequence_length, input_size, hidden_size_1, hidden_size_2, num_layers)

                best_accuracy_test = max(acc_per_epoch)
                if best_accuracy_test > best_acc_per_optimizer[opt_id]:
                    best_acc_per_optimizer[opt_id] = best_accuracy_test
                    best_epoch_per_optimizer[opt_id] =  acc_per_epoch.index(best_accuracy_test)
                    best_loss_per_optimizer[opt_id] = loss_per_epoch[best_epoch_per_optimizer[opt_id]]
                    best_learning_rate_per_optimizer[opt_id] = learning_rate
                    best_time_per_optimizer[opt_id] = trainingTime

            learning_rate = best_learning_rate_per_optimizer[opt_id]

            for batch_size in batch_sizes[1:]:
                (loss_per_epoch, acc_per_epoch, trainingTime, trainedModel) = train_pytorch_model(device, model_name, X_train, y_train, X_dev, y_dev, learning_rate, reg_constant, num_epochs_tune, batch_size, dropout_prob, arg_optimizer, sequence_length, input_size, hidden_size_1, hidden_size_2, num_layers)

                best_accuracy_test = max(acc_per_epoch)
                if best_accuracy_test > best_acc_per_optimizer[opt_id]:
                    best_acc_per_optimizer[opt_id] = best_accuracy_test
                    best_epoch_per_optimizer[opt_id] =  acc_per_epoch.index(best_accuracy_test)
                    best_loss_per_optimizer[opt_id] = loss_per_epoch[best_epoch_per_optimizer[opt_id]]
                    best_batch_size_per_optimizer[opt_id] = batch_size
                    best_time_per_optimizer[opt_id] = trainingTime

            batch_size = best_batch_size_per_optimizer[opt_id]

            for dropout_prob in dropout_probs[1:]:
                (loss_per_epoch, acc_per_epoch, trainingTime, trainedModel) = train_pytorch_model(device, model_name, X_train, y_train, X_dev, y_dev, learning_rate, reg_constant, num_epochs_tune, batch_size, dropout_prob, arg_optimizer, sequence_length, input_size, hidden_size_1, hidden_size_2, num_layers)

                best_accuracy_test = max(acc_per_epoch)
                if best_accuracy_test > best_acc_per_optimizer[opt_id]:
                    best_acc_per_optimizer[opt_id] = best_accuracy_test
                    best_epoch_per_optimizer[opt_id] =  acc_per_epoch.index(best_accuracy_test)
                    best_loss_per_optimizer[opt_id] = loss_per_epoch[best_epoch_per_optimizer[opt_id]]
                    best_dropout_prob_per_optimizer[opt_id] = dropout_prob
                    best_time_per_optimizer[opt_id] = trainingTime

            printing('Best results for %s optimizer (%s): (eta: %f, reg: %f, batch_size: %d, dropout_prob: %f, epoch_number: %d, loss: %.4f, accuracy: %.4f, time: %.2f s)' %(arg_optimizer, model_name, best_learning_rate_per_optimizer[opt_id], best_reg_per_optimizer[opt_id], best_batch_size_per_optimizer[opt_id], best_dropout_prob_per_optimizer[opt_id], best_epoch_per_optimizer[opt_id], best_loss_per_optimizer[opt_id], best_acc_per_optimizer[opt_id], best_time_per_optimizer[opt_id]))

        finishTuneTime = time.time()

        printing('\nTuning the hyperparameters has ended (duration:  {0:0.2f} seconds)'.format(finishTuneTime-finishLoadTime))

        best_acc = max(best_acc_per_optimizer)
        opt_id_aux = best_acc_per_optimizer.index(best_acc)
        best_optimizer = optimizers[opt_id_aux]
        best_learning_rate = best_learning_rate_per_optimizer[opt_id_aux]
        best_reg = best_reg_per_optimizer[opt_id_aux]
        best_batch_size = best_batch_size_per_optimizer[opt_id_aux]
        best_dropout_prob = best_dropout_prob_per_optimizer[opt_id_aux]
        best_epoch = best_epoch_per_optimizer[opt_id_aux]
        best_loss = best_loss_per_optimizer[opt_id_aux]
        best_time = best_time_per_optimizer[opt_id_aux]

        printing('Best results %s: (optimizer: %s, eta: %f, reg: %f, batch_size: %d, dropout_prob: %f, epoch_number: %d, loss: %.4f, accuracy: %.4f, time: %.2f s)' %(model_name, best_optimizer, best_learning_rate, best_reg, best_batch_size, best_dropout_prob, best_epoch, best_loss, best_acc, best_time))

        learning_rate = best_learning_rate
        reg_constant = best_reg
        num_epochs = best_epoch
        arg_optimizer = best_optimizer
        batch_size = best_batch_size
        dropout_prob = best_dropout_prob

    (loss_per_epoch, acc_per_epoch, trainingTime, trainedModel) = train_pytorch_model(device, model_name, X_train, y_train, X_dev, y_dev, learning_rate, reg_constant, num_epochs, batch_size, dropout_prob, arg_optimizer, sequence_length, input_size, hidden_size_1, hidden_size_2, num_layers)
    plotFiguresAccLoss(loss_per_epoch, acc_per_epoch, num_epochs, model_name, device, learning_rate, batch_size, arg_optimizer)

    printing('Evaluating on Testing Set...')
    (accuracy, loss) = evaluate_pytorch(device, model_name, trainedModel, X_test, y_test, sequence_length, input_size)
    printing('Test accuracy: %f' % accuracy)
    printing('Test Loss: %f' % loss)

    OF.close()

if __name__ == "__main__":
    main()
