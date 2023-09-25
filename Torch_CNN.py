import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.data.sampler import SubsetRandomSampler
import torchaudio
from Dataset_Generator import GTZANDataset
from ResNet_Model import ResNet50
from sklearn.model_selection import KFold
from My_Model import CNNNetwork
from VGG_Model import VGG16
from Metrics import plot_figures, metrics_extraction
import numpy as np

BATCH_SIZE = 40
EPOCHS = 6
LEARNING_RATE = 0.0001
ANNOTATIONS_FILE = "C:/Users/Stergios/Desktop/Datasets/GTZAN/annotations_file_3.csv"
AUDIO_DIR = "C:/Users/Stergios/Desktop/Datasets/GTZAN/genres_modified"
SAMPLE_RATE = 22050
NUM_SAMPLES = 3*22050
N_FOLDS = 5
TRAIN_SIZE = 0.7
RANDOM_SEED = 42

class_mapping_2 = [
   "classical", 
   "jazz",
   "rock"
]
class_mapping = [
    "blues",
    "classical",
    "counrty",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock"
]

def model_iter(model, loader, loss_func, data_type, device, optimizer, fold):
    totals_temp = []
    correct_temp = []
    loss_temp = []
    results = []
    avg_loss = []
    if data_type == "Validation":
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)

                predictions = model(inputs)

                loss = loss_func(predictions, targets)
                loss_temp.append(loss.item())

                # calculate correct and total predictions
                _, predicted = torch.max(predictions.data, 1)
                total = targets.size(0) 
                correct = (predicted == targets).sum().item()
            
                totals_temp.append(total)
                correct_temp.append(correct)
    
    else:
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            
            predictions = model(inputs)

            loss = loss_func(predictions, targets)
            loss_temp.append(loss.item())

            # backpropagate loss and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate correct and total predictions
            _, predicted = torch.max(predictions.data, 1)
            total = targets.size(0) 
            correct = (predicted == targets).sum().item()
            
            totals_temp.append(total)
            correct_temp.append(correct)

    total = sum(totals_temp)
    correct = sum(correct_temp)   
    results.append(100*(correct/total))
    total_loss = sum(loss_temp)
    avg_loss.append(total_loss/(len(loss_temp)))

    print(f"{data_type} Loss: {total_loss/len(loss_temp)}")
    
    # print val accuracy
    if data_type=="Validation":
        print(f"Validation Accuracy for fold {fold+1}: {100*(correct/total)} %")
        print("--------------------------")
        

    return results, avg_loss

def cross_validate(kfold, model, dataset, loss_fn, optimizer, device):
    results = []
    temp_acc = []
    temp_loss = []
    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []

    # Starting Cross Validation
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        

        train_subsampler = SubsetRandomSampler(train_ids)
        val_subsampler = SubsetRandomSampler(val_ids)

        train_data = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_subsampler)
        val_data = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=val_subsampler)
        
        # iterate for training set
        temp_acc, temp_loss = model_iter(model, train_data, loss_fn, "Training", device, optimizer, fold)
        train_acc.append(temp_acc)
        train_loss.append(temp_loss)

        # iterate for validation set
        temp_acc, temp_loss = model_iter(model, val_data, loss_fn, "Validation", device, optimizer, fold)
        val_acc.append(temp_acc)
        val_loss.append(temp_loss)

    # print kfold results    
    print(f"Results for K-Fold Cross Validation for {N_FOLDS} folds")
    print("--------------------------")
    
    val_acc=np.array(val_acc)

    for i in range(len(val_acc)):
        print(f"Fold {i+1}: {val_acc[i]} %")
    total_acc = sum(val_acc)
    print(f"Average: {total_acc/len(val_acc)} %")    

    return train_acc, train_loss, val_acc, val_loss

def train(kfold, model, dataset, loss_fn, optimizer, device, epochs):
    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []
    for i in range (epochs):
        print(f"Epoch {i+1}")
        print("--------------------------")
        temp_train_acc, temp_train_loss, temp_val_acc, temp_val_loss = cross_validate(kfold, model, dataset, loss_fn, optimizer, device)
        train_acc.append(temp_train_acc)
        val_acc.append(temp_val_acc)
        train_loss.append(temp_train_loss)
        val_loss.append(temp_val_loss)
        print("--------------------------")
        
    print("Training is done.")    
    return train_acc, train_loss, val_acc, val_loss

def test(model, test_set, device, train_acc, train_loss, val_acc, val_loss):
    print("Final Test")
    print("--------------------------")
    totals_temp = []
    right_temp = []
    
    test_data = DataLoader(test_set, batch_size=BATCH_SIZE)
    
    with torch.no_grad():
        for inputs, targets in test_data:
            inputs, targets = inputs.to(device), targets.to(device)

            predictions = model(inputs)

            _, predicted = torch.max(predictions.data, 1)
            total = targets.size(0) 
            correct = (predicted == targets).sum().item()
            
            totals_temp.append(total)
            right_temp.append(correct)

        total = sum(totals_temp)
        correct = sum(right_temp)
        print(f"Accuracy for test set is {100*(correct/total)} %")
        print("--------------------------")

    plot_figures(train_acc, val_acc, EPOCHS*N_FOLDS, "Accuracy")
    plot_figures(train_loss, val_loss, EPOCHS*N_FOLDS, "Loss")
    metrics_extraction(model, test_data, class_mapping)


if __name__ == "__main__":
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"    
    print(f"Using Device: {device}")

    # instantiating our dataset object and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, 
        n_fft=256,
        hop_length=128,
        n_mels=8
        )

    
    gtzan = GTZANDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE,  NUM_SAMPLES, device)


    # Split the dataset into train and test subsets
    train_length = int(len(gtzan)*TRAIN_SIZE) 
    test_length = len(gtzan) - train_length

    train_set, test_set = random_split(gtzan, [train_length, test_length], generator=torch.Generator().manual_seed(42))

    # get number of folds
    kfold = KFold(n_splits=N_FOLDS, shuffle=True)
    
    # construct model and assign it to device
    #model = ResNet50(num_classes=10, img_channel=1).to(device=device)
    model = VGG16().to(device=device)
    #model = CNNNetwork().to(device=device)
    print(model) 
    param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(param)
    # instantiate loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train_acc, train_loss, val_acc, val_loss = train(kfold, model, train_set, loss_fn, optimizer, device, EPOCHS)
    # final test
    test(model, test_set, device=device, train_acc=train_acc, train_loss=train_loss, val_acc=val_acc, val_loss=val_loss)

    # save the trained model
    torch.save(model.state_dict(), "8_Mels.pth")
    print("Model trained and stored at VGG16.pth")



