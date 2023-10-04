from numpy.core.function_base import linspace
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np

AUDIO_DIR = "<AUDIO_PATH>"
ANNOTATIONS_FILE = "<LABELS_PATH>"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050
BATCH_SIZE = 32

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

@torch.no_grad()
def get_all_preds(model, loader):
    all_preds = torch.tensor([]).to(device="cuda")
    all_labels = torch.tensor([]).to(device="cuda")
    for batch in loader:
        images, labels = batch
        images, labels = images.to(device="cuda"), labels.to(device="cuda")

        preds = model(images)
        all_preds = torch.cat(
            (all_preds, preds),
            dim=0
        )
        all_labels = torch.cat(
            (all_labels, labels),
            dim=0
        )

    return all_preds, all_labels

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()   

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization\n')

    print(cm)
    print("\n")
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def precision_recall(conf_matrix):
    # create precision and recall arrays
    new_mat = np.array(conf_matrix)
    precision = np.zeros(10)
    recall = np.zeros(10)
    f_measure = np.zeros(10)
    
    # loop calculating precision and recall from confusion matrix
    for i in range(len(conf_matrix)):
        recall[i] = new_mat[i, i]/sum(new_mat[i])
        precision[i] = new_mat[i, i]/sum(new_mat[:,i])
        f_measure[i] = 2*(precision[i]*recall[i]/(precision[i]+recall[i]))
    return precision, recall, f_measure    

def plot_metrics(table, class_map):
    fig, ax = plt.subplots()
    
    # hide axis
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    
    
    ax.table(cellText=table.T, loc="center", rowLabels=class_map, colLabels=["Precision", "Recall", "F-Measure"])
    fig.tight_layout()
    plt.show()

def metrics_extraction(model, loader, class_map):
    # get predictions and labels from test set
    train_preds, train_labels = get_all_preds(model, loader)

    # calculate accuracy
    preds_correct = get_num_correct(train_preds, train_labels)
    print('total correct:', preds_correct)
    print(f"accuracy:, {100*preds_correct / len(train_preds)} %")

    # calculate confusion matrix
    stacked = torch.stack(
    (train_labels, train_preds.argmax(dim=1)), dim=1
    )
    
    conf_matrix = torch.zeros(10,10, dtype=torch.int64)
    for p in stacked:
        tl, pl = p.tolist()
        tl, pl = int(tl), int(pl)
        conf_matrix[tl, pl] = conf_matrix[tl, pl] + 1

    # calculate precision, recall and f-measure
    precision, recall, f_measure = precision_recall(conf_matrix)

    table = np.vstack((precision, recall))
    table = np.vstack((table, f_measure))
    
    # plot results

    plot_confusion_matrix(conf_matrix, class_map)
    plot_metrics(table, class_map)

def plot_figures(train_fig, val_fig, num_epochs, ylabel):
    train_fig = np.array(train_fig)
    val_fig = np.array(val_fig)
    epochs = []
    train_fig = np.reshape(train_fig, num_epochs)
    val_fig = np.reshape(val_fig, num_epochs)
    for i in range(num_epochs):
        epochs.append(i+1)
    
    epochs = np.array(epochs)

    plt.figure(figsize=(18,15))
    
    plt.plot(epochs.T, train_fig)
    plt.plot(epochs.T, val_fig)

    plt.legend([f"Train {ylabel}", f"Validation {ylabel}"])
    plt.ylabel(f"{ylabel}")
    plt.xlabel("Epochs")
    plt.show()
    
if __name__=="__main__":

    cmat = (
        [252, 0, 4, 0, 1, 20, 0, 0, 15, 8],
        [1, 273, 1, 0, 0, 21, 0, 1, 1, 2], 
        [20, 1, 175, 4, 0, 23, 1, 8, 15, 53],
        [0, 4, 5, 243, 5, 1, 0, 21, 8, 13],
        [8, 2, 1, 10, 195, 0, 13, 20, 43, 8],
        [7, 17, 1, 1, 0, 262, 0, 5, 6, 1],
        [1, 0, 1, 0, 3, 3, 271, 1, 1, 19],
        [4, 2, 3, 2, 3, 4, 0, 271, 2, 9],
        [9, 2, 3, 10, 0, 9, 0, 5, 253, 9],
        [9, 7, 12, 9, 2, 9, 6, 13, 14, 219]
        )
    
    cmat = np.array(cmat)
    #cmat = cmat.T
    plot_confusion_matrix(cmat, classes=class_mapping)
    print(cmat.shape)
    pre, re, f = precision_recall(cmat)

    table = np.vstack((pre, re))
    table = np.vstack((table, f))
    plot_metrics(table ,class_map=class_mapping)
