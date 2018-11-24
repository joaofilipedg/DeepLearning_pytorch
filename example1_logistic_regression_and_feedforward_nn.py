import numpy as np
import matplotlib.pyplot as plt
import torch
import time

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
        x = np.array([float(v) for v in fields[6:]])
        if pairwise_features:
            x = x[:, None].dot(x[None, :]).flatten()
        X.append(x)
        y.append(k)
    f.close()
    l = ['' for k in labels]
    for letter in labels:
        l[labels[letter]] = letter
    return X, y, l

def evaluate_pytorch(device, model, X, y, model_name, activ_func='ReLU'):
    """Evaluate model on data."""

    test_x = torch.tensor(X, dtype=torch.float).to(device)
    test_y = torch.tensor(y, dtype=torch.long).to(device)

    loss_function = torch.nn.CrossEntropyLoss()

    if (model_name == 'nn_single'):
        y_pred = model(test_x, activ_func)
    else:
        y_pred = model(test_x)

    loss = loss_function(y_pred, test_y)
    avg_loss = loss / len(y)

    hits = torch.sum(y_pred.argmax(dim=1) == test_y).item()
    accuracy = float(hits) / len(y)

    return accuracy, loss

class logisticRegression(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(logisticRegression, self).__init__()
        self.fc = torch.nn.Linear(num_features, num_classes)
    def forward(self, x):
        return self.fc(x)

class singleNN(torch.nn.Module):
    def __init__(self, num_features, num_classes, dropout_prob):
        super(singleNN, self).__init__()
        self.fc1 = torch.nn.Linear(num_features, num_classes)
        self.act_sigm = torch.nn.Sigmoid()
        self.act_tanh = torch.nn.Tanh()
        self.act_relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout_prob)

    def forward(self, x, activ):
        out = self.fc1(x)
        if (activ == 'sigmoid'):
            out = self.act_sigm(out)
        elif (activ == 'tanh'):
            out = self.act_tanh(out)
        else:
            out = self.act_relu(out)
        out = self.dropout(out)
        return out

class multiNN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_size_1, hidden_size_2, dropout_prob):
        super(multiNN, self).__init__()
        self.fc1 = torch.nn.Linear(num_features, hidden_size_1)
        self.act_relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.fc2 = torch.nn.Linear(hidden_size_1, hidden_size_2)
        self.dropout2 = torch.nn.Dropout(dropout_prob)
        self.fc3 = torch.nn.Linear(hidden_size_2, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.act_relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.act_relu(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        return out

def plotFiguresAccLoss(loss_per_epoch, acc_per_epoch, num_epochs, model_name, device, reg, learning_rate, batch_size, dropout_prob=0, activ_func='sigmoid', optimizer='SGD'):
    printing('FigureACCLossFunc: model name: %s, optimizer: %s' %(model_name, optimizer))
    plt.figure(1)
    plt.plot(range(1, num_epochs+1), loss_per_epoch, 'bo-')
    plt.title('Dev Losses')
    name = 'loss_%s_%s_%s_%s_%s' % (device, model_name, reg, learning_rate, batch_size)
    name += '_pairwise'
    if not (model_name == 'logistic'):
        name += '_%s_%s_%s' %(dropout_prob, activ_func, optimizer)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig('%s.pdf' % name)
    plt.close()

    plt.figure(2)
    plt.plot(range(1, num_epochs+1), acc_per_epoch, 'bo-')
    plt.title('Dev Accuracy')
    name = 'acc_%s_%s_%s_%s_%s' % (device, model_name, reg, learning_rate, batch_size)
    name += '_pairwise'
    if not (model_name == 'logistic'):
        name += '_%s_%s_%s' %(dropout_prob, activ_func, optimizer)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.savefig('%s.pdf' % name)
    plt.close()

def checkBestAccuracy(acc_per_epoch, loss_per_epoch, reg, batch_size, learning_rate, trainingTime, dropout_prob, best_acc, best_epoch, best_loss, best_reg, best_batch_size, best_learning_rate, best_time, best_dropout_prob):
    best_accuracy_test = max(acc_per_epoch)
    if (best_accuracy_test > best_acc):
        best_acc = best_accuracy_test
        best_epoch =  acc_per_epoch.index(best_accuracy_test)
        best_loss = loss_per_epoch[best_epoch]
        best_reg = reg
        best_batch_size = batch_size
        best_learning_rate = learning_rate
        best_time = trainingTime
        best_dropout_prob = dropout_prob

    return best_acc, best_epoch, best_loss, best_reg, best_batch_size, best_learning_rate, best_time, best_dropout_prob

def train_pytorch_model(device, model_name, X_train, y_train, X_dev, y_dev, learning_rate, reg, num_epochs, batch_size, dropout_prob=0, activ_func='sigmoid', optimizer_name='SGD'):
    num_features = 128*128
    num_classes = 26

    trainTime = time.time()
    if (model_name == 'logistic'):
        printing('Training %s model(eta: %f, reg: %f, num_epochs: %d, batch_size: %d)...' % (model_name, learning_rate, reg, num_epochs, batch_size))
    else:
        printing('Training %s model(eta: %f, reg: %f, num_epochs: %d, batch_size: %d, dropout_prob: %f, activation_func: %s, optimizer: %s)...' % (model_name, learning_rate, reg, num_epochs, batch_size, dropout_prob, activ_func, optimizer_name))

    if (device == 'cpu'):
        torch.manual_seed(42)
    else:
        torch.cuda.manual_seed(42)

    if (model_name == 'logistic'):
        model = logisticRegression(num_features, num_classes).to(device)
    elif (model_name == 'nn_single'):
        model = singleNN(num_features, num_classes, dropout_prob).to(device)
    else:
        hidden_size_1 = 100
        hidden_size_2 = 100
        model = multiNN(num_features, num_classes, hidden_size_1, hidden_size_2, dropout_prob).to(device)

    loss_function = torch.nn.CrossEntropyLoss()

    optimizer_SGD = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer_Adam = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer_Adagrad= torch.optim.Adagrad(model.parameters(), lr=learning_rate)

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

        while batch_index < len(X_train):
            # get the data for this batch
            next_index = batch_index + batch_size
            batch_x = torch.tensor(X_train[batch_index:next_index], dtype=torch.float).to(device)
            batch_labels = torch.tensor(y_train[batch_index:next_index], dtype=torch.long).to(device)
            batch_index = next_index

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            if (model_name == 'nn_single'):
                logits = model(batch_x, activ_func)
            else:
                logits = model(batch_x)

            loss = loss_function(logits, batch_labels)
            loss_value = loss.item()
            total_loss += loss_value
            losses.append(loss_value)

            loss.backward()
            optimizer.step()

        #evaluate on evaluation dataset
        (acc, avg_loss) = evaluate_pytorch(device, model, X_dev, y_dev, model_name, activ_func)

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
    parser.add_argument('--model', type=str, default="logistic")
    parser.add_argument('--activ_func', type=str, default="sigmoid")
    parser.add_argument('--optimizer', type=str, default="SGD")
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--reg_constant', type=float, default=0.0)
    parser.add_argument('--dropout_prob', type=float, default=0.0)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--tune', action='store_const', const=True, default=False)

    args = vars(parser.parse_args())
    printing(args)

    device_arg = args['device']
    num_epochs = args['num_epochs']
    device_id = args['device_id']
    model_name = args['model']
    arg_optimizer = args['optimizer']
    arg_batch_size = args['batch_size']
    arg_learning_rate = args['learning_rate']
    arg_reg_constant = args['reg_constant']
    arg_dropout_prob = args['dropout_prob']
    arg_activ_func = args['activ_func']
    tune = args['tune']

    pairwise=True

    filepath = 'ocr_dataset.data'
    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [1, 4, 8]
    regs = [0, 0.1, 0.01, 0.001, 0.0001]
    dropout_probs = [0, 0.25]
    activ_functions = ['sigmoid', 'relu', 'tanh']
    optimizers = ['SGD', 'Adam', 'Adagrad']

    best_acc = 0
    best_epoch =  -1
    best_loss = -1.0
    best_reg = -1.0
    best_batch_size = -1
    best_learning_rate = -1.0
    best_time = -1.0
    best_dropout_prob = -1.0

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

    if (model_name == 'logistic'):
        if (tune == True):
            #tune hyperparameters with a grid search
            for reg in regs:
                for learning_rate in learning_rates:
                    for batch_size in batch_sizes:
                        (loss_per_epoch, acc_per_epoch, trainingTime, trainedModel) = train_pytorch_model(device, model_name, X_train, y_train, X_dev, y_dev, learning_rate, reg, num_epochs, batch_size)

                        #check if it is the best accuracy (on validation set) found yet
                        (best_acc, best_epoch, best_loss, best_reg, best_batch_size, best_learning_rate, best_time, best_dropout_prob_notused) = checkBestAccuracy(acc_per_epoch, loss_per_epoch, reg, batch_size, learning_rate, trainingTime, 0, best_acc, best_epoch, best_loss, best_reg, best_batch_size, best_learning_rate, best_time, best_dropout_prob)
            printing('Best results %s: (eta: %f, reg: %f, batch_size: %d, epoch_number: %d, loss: %.4f, accuracy: %.4f, time: %.2f s)' %(model_name, best_learning_rate, best_reg, best_batch_size, best_epoch, best_loss, best_acc, best_time))
        else:
            best_reg = arg_reg_constant
            best_batch_size = arg_batch_size
            best_learning_rate = arg_learning_rate
            best_epoch = num_epochs

        (loss_per_epoch, acc_per_epoch, trainingTime, trainedModel) = train_pytorch_model(device, model_name, X_train, y_train,  X_dev, y_dev, best_learning_rate, best_reg, best_epoch, best_batch_size)

        plotFiguresAccLoss(loss_per_epoch, acc_per_epoch, best_epoch, model_name, device, best_reg, best_learning_rate, best_batch_size)

        printing('Evaluating...')
        (accuracy, loss) = evaluate_pytorch(device, trainedModel, X_test, y_test, model_name)
        printing('Test accuracy: %f' % accuracy)
        printing('Test Loss: %f' % loss)

    elif (model_name == 'nn_single'):
        if (tune == True):
            #grid search would be too time consuming here
            for activ_func in activ_functions:
                for optimizer in optimizers:
                    best_acc = 0.0
                    batch_size = batch_sizes[1]
                    learning_rate = learning_rates[0]
                    dropout_prob = dropout_probs[0]
                    for reg in regs:
                        (loss_per_epoch, acc_per_epoch, trainingTime, trainedModel) = train_pytorch_model(device, model_name, X_train, y_train, X_dev, y_dev, learning_rate, reg, num_epochs, batch_size, dropout_prob, activ_func, optimizer)
                        printing('Main: model name: %s, optimizer: %s' %(model_name, optimizer))

                        #check if it is the best accuracy (on validation set) found yet
                        (best_acc, best_epoch, best_loss, best_reg, best_batch_size, best_learning_rate,  best_time, best_dropout_prob) = checkBestAccuracy(acc_per_epoch, loss_per_epoch, reg, batch_size, learning_rate, trainingTime, dropout_prob, best_acc, best_epoch, best_loss, best_reg, best_batch_size, best_learning_rate, best_time, best_dropout_prob)

                        printing('Best reg value: %f (eta: %f, reg: %f, batch_size: %d, dropout_prob: %f, epoch_number: %d, loss: %.4f, accuracy: %.4f, time: %.2f s)' %(best_reg, best_learning_rate, best_reg, best_batch_size, best_dropout_prob, best_epoch, best_loss, best_acc, best_time))

                    for learning_rate in learning_rates[1:]:
                        (loss_per_epoch, acc_per_epoch, trainingTime, trainedModel) = train_pytorch_model(device, model_name, X_train, y_train, X_dev, y_dev, learning_rate, best_reg, num_epochs, batch_size, dropout_prob, activ_func, optimizer)
                        printing('Main: model name: %s, optimizer: %s' %(model_name, optimizer))

                        #check if it is the best accuracy (on validation set) found yet
                        (best_acc, best_epoch, best_loss, best_reg, best_batch_size, best_learning_rate,  best_time, best_dropout_prob) = checkBestAccuracy(acc_per_epoch, loss_per_epoch, best_reg, batch_size, learning_rate, trainingTime, dropout_prob, best_acc, best_epoch, best_loss, best_reg, best_batch_size, best_learning_rate, best_time, best_dropout_prob)

                        printing('Best learning_rate value: %f (eta: %f, reg: %f, batch_size: %d, dropout_prob: %f, epoch_number: %d, loss: %.4f, accuracy: %.4f, time: %.2f s)' %(best_learning_rate, best_learning_rate, best_reg, best_batch_size, best_dropout_prob, best_epoch, best_loss, best_acc, best_time))

                    for dropout_prob in dropout_probs[1:]:
                        (loss_per_epoch, acc_per_epoch, trainingTime, trainedModel) = train_pytorch_model(device, model_name, X_train, y_train, X_dev, y_dev, best_learning_rate, best_reg, num_epochs, best_batch_size, dropout_prob, activ_func, optimizer)
                        printing('Main: model name: %s, optimizer: %s' %(model_name, optimizer))

                        #check if it is the best accuracy (on validation set) found yet
                        (best_acc, best_epoch, best_loss, best_reg, best_batch_size, best_learning_rate,  best_time, best_dropout_prob) = checkBestAccuracy(acc_per_epoch, loss_per_epoch, best_reg, best_batch_size, best_learning_rate, trainingTime, dropout_prob, best_acc, best_epoch, best_loss, best_reg, best_batch_size, best_learning_rate, best_time, best_dropout_prob)

                        printing('Best dropout_prob value: %f (eta: %f, reg: %f, batch_size: %d, dropout_prob: %f, epoch_number: %d, loss: %.4f, accuracy: %.4f, time: %.2f s)' %(best_dropout_prob, best_learning_rate, best_reg, best_batch_size, best_dropout_prob, best_epoch, best_loss, best_acc, best_time))
        else:
            best_reg = arg_reg_constant
            best_learning_rate = arg_learning_rate
            best_batch_size = arg_batch_size
            best_dropout_prob = arg_dropout_prob
            best_reg = arg_reg_constant
            activ_func = arg_activ_func
            optimizer = arg_optimizer
            best_epoch = num_epochs

        (loss_per_epoch, acc_per_epoch, trainingTime, trainedModel) = train_pytorch_model(device, model_name, X_train, y_train, X_dev, y_dev, best_learning_rate, best_reg, best_epoch, best_batch_size, best_dropout_prob, activ_func, optimizer)

        plotFiguresAccLoss(loss_per_epoch, acc_per_epoch, best_epoch, model_name, device, best_reg, best_learning_rate, best_batch_size, best_dropout_prob, activ_func, optimizer)

        printing('Evaluating...')
        (accuracy, loss) = evaluate_pytorch(device, trainedModel, X_test, y_test, model_name, arg_activ_func)
        printing('Test accuracy: %f' % accuracy)
        printing('Test Loss: %f' % loss)

    elif (model_name == 'nn_multi'):
        print('Creating a neural network with 2 hidden layers (sizes 100 and 100), with %s activation functions, %s optimizer and dropout probability of %f' %(arg_activ_func, arg_optimizer, dropout_prob))

        (loss_per_epoch, acc_per_epoch, trainingTime, trainedModel) = train_pytorch_model(device, model_name, X_train, y_train, X_dev, y_dev, arg_learning_rate, arg_reg_constant, num_epochs, arg_batch_size, arg_dropout_prob, arg_activ_func, arg_optimizer)

        plotFiguresAccLoss(loss_per_epoch, acc_per_epoch, num_epochs, model_name, device, arg_reg_constant, arg_learning_rate, arg_batch_size, arg_dropout_prob, arg_activ_func, arg_optimizer)

        printing('Evaluating...')
        (accuracy, loss) = evaluate_pytorch(device, trainedModel, X_test, y_test, model_name)
        printing('Test accuracy: %f' % accuracy)
        printing('Test Loss: %f' % loss)
    else:
        printing('Wrong Model name')
        sys.exit()

    OF.close()
if __name__ == "__main__":
    main()
