from images import get_images
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
from constants import *
from dataset import ImageClassifierDataset
from utils import random_subset


class Unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unit, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=3, out_channels=out_channels, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, tensor_input):
        output = self.conv(tensor_input)
        output = self.bn(output)
        output = self.relu(output)

        return output


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()

        # Create 14 layers of the unit with max pooling in between
        self.unit1 = Unit(in_channels=3, out_channels=PIC_SIZE)
        self.unit2 = Unit(in_channels=PIC_SIZE, out_channels=PIC_SIZE)
        self.unit3 = Unit(in_channels=PIC_SIZE, out_channels=PIC_SIZE)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.unit4 = Unit(in_channels=PIC_SIZE, out_channels=PIC_SIZE * 2)
        self.unit5 = Unit(in_channels=PIC_SIZE * 2, out_channels=PIC_SIZE * 2)
        self.unit6 = Unit(in_channels=PIC_SIZE * 2, out_channels=PIC_SIZE * 2)
        self.unit7 = Unit(in_channels=PIC_SIZE * 2, out_channels=PIC_SIZE * 2)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.unit8 = Unit(in_channels=PIC_SIZE * 2, out_channels=PIC_SIZE * 4)
        self.unit9 = Unit(in_channels=PIC_SIZE * 4, out_channels=PIC_SIZE * 4)
        self.unit10 = Unit(in_channels=PIC_SIZE * 4, out_channels=PIC_SIZE * 4)
        self.unit11 = Unit(in_channels=PIC_SIZE * 4, out_channels=PIC_SIZE * 4)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.unit12 = Unit(in_channels=PIC_SIZE * 4, out_channels=PIC_SIZE * 4)
        self.unit13 = Unit(in_channels=PIC_SIZE * 4, out_channels=PIC_SIZE * 4)
        self.unit14 = Unit(in_channels=PIC_SIZE * 4, out_channels=PIC_SIZE * 4)

        self.avg_pool = nn.AvgPool2d(kernel_size=4)

        # Add all the units into the Sequential layer in exact order
        self.net = nn.Sequential(self.unit1, self.unit2, self.unit3, self.pool1, self.unit4, self.unit5, self.unit6,
                                 self.unit7, self.pool2, self.unit8, self.unit9, self.unit10, self.unit11, self.pool3,
                                 self.unit12, self.unit13, self.unit14, self.avg_pool)

        self.fc = nn.Linear(in_features=16 * 64, out_features=2)

    def forward(self, tensor_input):
        output = self.net(tensor_input)
        output = output.view(-1, 16 * 64)
        output = self.fc(output)
        return output


# Create a learning rate adjustment function that divides the learning rate by 10 every 30 epochs
def adjust_learning_rate(epoch):
    lr = 0.01

    if epoch > 180:
        lr = lr / 100000
    elif epoch > 150:
        lr = lr / 10000
    elif epoch > 120:
        lr = lr / 1000
    elif epoch > 90:
        lr = lr / 100
    elif epoch > 60:
        lr = lr / 10
    elif epoch > 30:
        lr = lr / 3

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def save_models(epoch):
    torch.save(model.state_dict(), f"cifar10model_{epoch}.model")
    print("Checkpoint saved")


def test(test_loader):
    model.eval()
    test_acc = 0.0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i, (images, labels) in enumerate(test_loader):

        if cuda_avail:
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        # Predict classes using images from the test set
        outputs = model(images)
        # prediction = prediction.cpu().numpy()
        _, prediction = torch.max(outputs.data, 1)

        # 0 means vehicle was detected so 0 == True
        for j in range(len(prediction)):
            if prediction[j] == labels.data[j] == 0:
                TP += 1
            elif prediction[j] == labels.data[j] == 1:
                TN += 1
            elif prediction[j] == 0 and labels.data[j] == 1:
                FP += 1
            elif prediction[j] == 1 and labels.data[j] == 0:
                FN += 1
            else:
                print('BIG error')
        test_acc += torch.sum(torch.eq(prediction, labels.data))

    # Compute the average acc and loss over all 30 test images
    test_acc = test_acc / 30
    print("correct accuracy", test_acc)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = 0
    recall = 0
    if TP + FP != 0:
        precision = TP / (TP + FP)
    if TP + FN != 0:
        recall = TP / (TP + FN)
    # Compute the average acc and loss over all 30 test images
    return accuracy, precision, recall


def train(num_epochs, train_loader, test_loader):
    print("Training~~")
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_acc = 0.0
        train_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            # Move images and labels to gpu if available
            if cuda_avail:
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())

            # Clear all accumulated gradients
            optimizer.zero_grad()
            # Predict classes using images from the test set
            outputs = model(images)
            # Compute the loss based on the predictions and actual labels

            loss = loss_fn(outputs, labels)
            # Backpropagation of the loss
            loss.backward()

            # Adjust parameters according to the computed gradients
            optimizer.step()

            train_loss += loss.cpu().data.item() * images.size(0)
            _, prediction = torch.max(outputs.data, 1)

            train_acc += torch.sum(prediction == labels.data)

        # Call the learning rate adjustment function
        adjust_learning_rate(epoch)

        # Compute the average acc and loss over all 120 training images
        train_acc = train_acc / 120
        train_loss = train_loss / 120

        # Evaluate on the test set
        accuracy, precision, recall = test(test_loader)

        # Save the model if the test acc is greater than our current best
        if accuracy > best_acc:
            save_models(epoch)
            best_acc = accuracy

        # Print the metrics
        print(
            f"Epoch {epoch: <3}, Train Accuracy: {train_acc: <20}, TrainLoss: {train_loss: <20}, Test Accuracy: {accuracy: <20}, Test Precision: {precision: <20}, Test Recall: {recall: <20}")


if __name__ == "__main__":
    # Define transformations for the training set, flip the images randomly, crop out and apply mean and std normalization
    train_transformations = transforms.Compose([
        transforms.Resize(PIC_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(PIC_SIZE, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transformations = transforms.Compose([
        transforms.Resize(PIC_SIZE),
        transforms.CenterCrop(PIC_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Check if gpu support is available
    cuda_avail = torch.cuda.is_available()

    # Create model, optimizer and loss function
    model = SimpleNet()

    if cuda_avail:
        model.cuda()

    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=0.001)
    loss_fn = nn.CrossEntropyLoss()

    all_images = get_images()
    cars = [(image, 0) for image in all_images.vehicle_images]
    randoms = [(image, 1) for image in all_images.other_images]

    train_cars, test_cars = random_subset(cars, .8)
    train_randoms, test_randoms = random_subset(randoms, .8)

    train_images = train_cars + train_randoms
    test_images = test_cars + test_randoms

    # train_set = image_classifier
    train_set = ImageClassifierDataset(
        train_images,
        [0, 1],
        train_transformations
    )

    # Create a loader for the training set
    train_dl = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    test_set = ImageClassifierDataset(
        test_images,
        [0, 1],
        test_transformations
    )

    # Create a loader for the test set, note that both shuffle is set to false for the test loader
    test_dl = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    train(EPOCHS, train_dl, test_dl)
