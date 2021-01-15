import os
import sys
import torch
import torch.nn.functional as nn
from torch.utils.data import SubsetRandomSampler
from gcommand_dataset import GCommandLoader


def main():
    batch = 64
    epoches = 10
    learning_rate = 0.001

    train_set = GCommandLoader("gcommands/train")
    validation_set = GCommandLoader("gcommands/valid")
    test_set = GCommandLoader("gcommands/test")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch, shuffle=True, pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set)

    model = CNN_Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for i in range(epoches):
        print("epoch: " + str(i))
        train(model, train_loader, optimizer)
        print("finished train")
        validation(model, validation_loader)
        print("finished validation")
    predict_test_y(model, test_loader, train_set, test_set)


class CNN_Model(torch.nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.layer1 = self.build_layer(1, 8, 5, 2, 1, False)
        self.layer2 = self.build_layer(8, 16, 3, 1, 2, True)
        self.layer3 = self.build_layer(16, 32, 3, 1, 2, False)
        self.layer4 = self.build_layer(32, 48, 3, 1, 2, True)
        self.layer5 = self.build_layer(48, 64, 3, 1, 2, False)
        self.layer6 = self.build_layer(64, 128, 3, 1, 2, True)
        self.layer7 = self.build_layer(128, 128, 3, 1, 2, False)
        self.layer8 = self.build_layer(128, 128, 3, 1, 2, True)
        self.layer9 = self.build_layer(128, 128, 3, 1, 2, True)
        self.fc0 = self.build_fc_layer(2560, 512)
        self.fc1 = self.build_fc_layer(512, 30)
        self.drop_out = torch.nn.Dropout()

    def build_layer(self, start, end, kernel_param, stride_param, pad_param, do_pool):
        if do_pool:
            return torch.nn.Sequential(
                torch.nn.Conv2d(start, end, kernel_size=kernel_param, stride=stride_param, padding=pad_param),
                torch.nn.ReLU(), torch.nn.BatchNorm2d(end), torch.nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            return torch.nn.Sequential(
                torch.nn.Conv2d(start, end, kernel_size=kernel_param, stride=stride_param, padding=pad_param),
                torch.nn.ReLU(), torch.nn.BatchNorm2d(end))

    def build_fc_layer(self, start, end):
        return torch.nn.Sequential(torch.nn.Linear(start, end), torch.nn.ReLU(), torch.nn.BatchNorm1d(end))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = x.view(x.size(0), -1)
        # x = self.drop_out(x)
        x = self.fc0(x)
        x = self.fc1(x)
        return nn.log_softmax(x, dim=1)


def train(model, train_loader, optimizer):
    model.train()
    for curr_batch, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = nn.nll_loss(output, labels)
        loss.backward()
        optimizer.step()


def validation(model, validation_loader):
    batch = 64
    model.eval()
    accuracy_counter = 0
    with torch.no_grad():
        for data, target in validation_loader:
            output = model(data)  # forward.
            # sum up batch loss and get the index of the max log-probability.
            # validation_loss += nn.nll_loss(output, target, reduction='sum').item()
            prediction = output.max(1, keepdim=True)[1]
            accuracy_counter += prediction.eq(target.view_as(prediction)).cpu().sum()

    accuracy_counter = 100. * accuracy_counter / (len(validation_loader) * batch)
    print(accuracy_counter)


def predict_test_y(model, test_loader, train_set, test_set):
    model.eval()
    classes = train_set.classes
    files_name = test_set.spects
    i = 0
    test_y = open('test_y', 'w')
    pred_list = []
    for test in test_loader:
        output = model(test[0])
        _, batch_pred = torch.max(output.data, 1)
        batch_pred = batch_pred.tolist()
        for predict in batch_pred:
            name = classes[predict]
            file_name = os.path.basename(files_name[i][0])
            pred_list.append("{},{}\n".format(file_name, name))
            i += 1
    pred_list = sorted(pred_list, key=lambda x: int(x.split('.')[0]))
    for predict in pred_list:
        test_y.write(predict)
    test_y.close()


if __name__ == "__main__":
    main()