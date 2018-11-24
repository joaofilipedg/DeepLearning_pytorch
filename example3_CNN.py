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

class ConvNet(torch.nn.Module):
    def __init__(self, dropout_prob, num_classes=26):
        super(ConvNet, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(20, 30, kernel_size=7, stride=1, padding=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = torch.nn.Dropout(dropout_prob)
        self.fc = torch.nn.Linear(30*4*2, num_classes)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc(out)
        out = torch.nn.functional.log_softmax(out, dim=1)
        return out

def plotFiguresAccLoss(loss_per_epoch, acc_per_epoch, num_epochs, model_name, device, learning_rate, reg_constant, batch_size, dropout_prob, optimizer):
    printing('FigureACCLossFunc: model name: %s, optimizer: %s' %(model_name, optimizer))
    plt.figure(1)
    plt.plot(range(1, num_epochs+1), loss_per_epoch, 'bo-')
    plt.title('Dev Losses')
    name = 'loss_%s_%s_%s_%s_%s_%s_%s' % (device, model_name, learning_rate, reg_constant, batch_size, dropout_prob, optimizer)

    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig('%s.pdf' % name)
    plt.close()

    plt.figure(2)
    plt.plot(range(1, num_epochs+1), acc_per_epoch, 'bo-')
    plt.title('Dev Accuracy')
    name = 'acc_%s_%s_%s_%s_%s_%s_%s' % (device, model_name, learning_rate, reg_constant, batch_size, dropout_prob, optimizer)

    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.savefig('%s.pdf' % name)
    plt.close()

def plotFilterVisual(trainedModel, device, model_name, learning_rate, reg_constant, batch_size, dropout_prob, optimizer):
    model_weights = trainedModel.state_dict()
    print(model_weights.keys())
    tensor_weight_example = model_weights['layer1.0.weight'][0].to(torch.device("cpu"))
    print(tensor_weight_example.numpy()[0])

    fig1, axs = plt.subplots(nrows=2,ncols=3)
    fig1.suptitle('CNN Filters Visualization')

    indexes=np.zeros([2,3], dtype=np.int32)
    indexes[0] = np.random.randint(20,size=3)
    indexes[1] = np.random.randint(30,size=3)
    print(indexes)

    for layer_id in range(2):
        for channel_id in range(3):
            if (layer_id == 0):
                tensor_layer = model_weights['layer1.0.weight']
            else:
                tensor_layer = model_weights['layer2.0.weight']
            tensor_layer_cpu = tensor_layer[indexes[layer_id][channel_id]].to(torch.device("cpu"))
            axs[layer_id][channel_id].set_title('Layer: %d; Channel: %d' %(layer_id+1, indexes[layer_id][channel_id]))
            axs[layer_id][channel_id].imshow(tensor_layer_cpu.numpy()[0])

    name = 'filters_%s_%s_%s_%s_%s_%s_%s' % (device, model_name, learning_rate, reg_constant, batch_size, dropout_prob, optimizer)
    plt.savefig('%s.pdf' % name)
    plt.close()

def evaluate_pytorch(device, model, X, y):
    """Evaluate model on data."""

    test_x = torch.tensor(X, dtype=torch.float).to(device)
    test_y = torch.tensor(y, dtype=torch.long).to(device)

    loss_function = torch.nn.CrossEntropyLoss()

    y_pred = model(test_x)

    loss = loss_function(y_pred, test_y)
    loss_value = loss.item()
    avg_loss = loss_value / len(y)

    hits = torch.sum(y_pred.argmax(dim=1) == test_y).item()
    accuracy = float(hits) / len(y)

    return accuracy, loss_value

def train_pytorch_model(device, model_name, X_train, y_train, X_dev, y_dev, learning_rate, reg, num_epochs, batch_size, dropout_prob, optimizer_name='SGD'):
    num_features = 16*8
    num_classes = 26

    if (model_name == 'ConvNet'):
        printing('Creating a Convolutional Neural Network with 2 layers')
    else:
        printing('Wrong Model name')
        sys.exit()

    trainTime = time.time()
    printing('Training %s model(eta: %f, reg_constant: %f, dropout_prob: %f, num_epochs: %d, batch_size: %d, optimizer: %s)...' % (model_name, learning_rate, reg, dropout_prob, num_epochs, batch_size, optimizer_name))

    if (device == 'cpu'):
        torch.manual_seed(42)
    else:
        torch.cuda.manual_seed(42)

    model = ConvNet(dropout_prob, num_classes).to(device)

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

            if (batch_index == 0):
                print(batch_x.size())

            batch_labels = torch.tensor(y_train[batch_index:next_index], dtype=torch.long).to(device)
            batch_index = next_index

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_function(logits, batch_labels)
            loss_value = loss.item()
            total_loss += loss_value
            losses.append(loss_value)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss_value
            if batch_index % 999*batch_size == 0:    # print every 2000 mini-batches
                printing('[%d, %5d] loss: %.3f' % (epoch + 1, batch_index + 1, running_loss / 1000))
                running_loss = 0.0

        #evaluate on evaluation dataset
        (acc, avg_loss) = evaluate_pytorch(device, model, X_dev, y_dev)

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
    parser.add_argument('--model', type=str, default="ConvNet")
    parser.add_argument('--optimizer', type=str, default="SGD")
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

    filepath = 'ocr_dataset.data'

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
                (loss_per_epoch, acc_per_epoch, trainingTime, trainedModel) = train_pytorch_model(device, model_name, X_train, y_train, X_dev, y_dev, learning_rate, reg_constant, num_epochs_tune, batch_size, dropout_prob, arg_optimizer)

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
                (loss_per_epoch, acc_per_epoch, trainingTime, trainedModel) = train_pytorch_model(device, model_name, X_train, y_train, X_dev, y_dev, learning_rate, reg_constant, num_epochs_tune, batch_size, dropout_prob, arg_optimizer)

                best_accuracy_test = max(acc_per_epoch)
                if best_accuracy_test > best_acc_per_optimizer[opt_id]:
                    best_acc_per_optimizer[opt_id] = best_accuracy_test
                    best_epoch_per_optimizer[opt_id] =  acc_per_epoch.index(best_accuracy_test)
                    best_loss_per_optimizer[opt_id] = loss_per_epoch[best_epoch_per_optimizer[opt_id]]
                    best_learning_rate_per_optimizer[opt_id] = learning_rate
                    best_time_per_optimizer[opt_id] = trainingTime

            learning_rate = best_learning_rate_per_optimizer[opt_id]

            for batch_size in batch_sizes[1:]:
                (loss_per_epoch, acc_per_epoch, trainingTime, trainedModel) = train_pytorch_model(device, model_name, X_train, y_train, X_dev, y_dev, learning_rate, reg_constant, num_epochs_tune, batch_size, dropout_prob, arg_optimizer)

                best_accuracy_test = max(acc_per_epoch)
                if best_accuracy_test > best_acc_per_optimizer[opt_id]:
                    best_acc_per_optimizer[opt_id] = best_accuracy_test
                    best_epoch_per_optimizer[opt_id] =  acc_per_epoch.index(best_accuracy_test)
                    best_loss_per_optimizer[opt_id] = loss_per_epoch[best_epoch_per_optimizer[opt_id]]
                    best_batch_size_per_optimizer[opt_id] = batch_size
                    best_time_per_optimizer[opt_id] = trainingTime

            batch_size = best_batch_size_per_optimizer[opt_id]

            for dropout_prob in dropout_probs[1:]:
                (loss_per_epoch, acc_per_epoch, trainingTime, trainedModel) = train_pytorch_model(device, model_name, X_train, y_train, X_dev, y_dev, learning_rate, reg_constant, num_epochs_tune, batch_size, dropout_prob, arg_optimizer)

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

    (loss_per_epoch, acc_per_epoch, trainingTime, trainedModel) = train_pytorch_model(device, model_name, X_train, y_train, X_dev, y_dev, learning_rate, reg_constant, num_epochs, batch_size, dropout_prob, arg_optimizer)
    plotFiguresAccLoss(loss_per_epoch, acc_per_epoch, num_epochs, model_name, device, learning_rate, reg_constant, batch_size, dropout_prob, arg_optimizer)

    plotFilterVisual(trainedModel, device, model_name, learning_rate, reg_constant, batch_size, dropout_prob, arg_optimizer)

    printing('Evaluating on Testing Set...')
    (accuracy, loss) = evaluate_pytorch(device, trainedModel, X_test, y_test)
    printing('Test accuracy: %f' % accuracy)
    printing('Test Loss: %f' % loss)

    OF.close()

if __name__ == "__main__":
    main()
