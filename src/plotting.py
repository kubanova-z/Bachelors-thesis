import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


#CONFUSION MATRIX
def plot_confusion_matrix(true_ids, preds_ids, target_names, prefix=''):
    cm = confusion_matrix(true_ids, preds_ids)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    #display object
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=target_names)

    #plot matrix
    fig, ax = plt.subplots(figsize=(10,10))
    disp.plot(
        cmap=plt.cm.PuBuGn, 
        ax=ax, 
        xticks_rotation='vertical',
        values_format='.1%')

    ax.set_title('Confusion matrix', fontsize=16, fontweight='bold')
    ax.set_ylabel('True Category', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Category', fontsize=12, fontweight='bold')
    
    plt.tight_layout()

    #plt.show()
    filename = f'{prefix}confusion_matrix_results.png'
    plt.savefig(filename) 
    plt.close(fig) 
    
    


#LEARNING CURVE
def plot_learning_curve(epochs, train_loss, test_acc, prefix=''):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
    epoch_range = range(1, epochs+1)

    ax1.plot(epoch_range, train_loss, label='Training loss', color='red')
    ax1.set_title('Training loss per epoch', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (Cross Entropy)', fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)


    ax2.plot(epoch_range, test_acc, label='Test accuracy', color='blue')
    ax2.set_title('Test accuracy per Epoch', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    filename = f'{prefix}learning_curve.png'

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)

    
# bar chart - F1 score
def plot_metrics_bar_chart(true_ids, preds_ids, target_names, prefix=''):

    #classification report as dictionary
    report = classification_report(
        true_ids, preds_ids, target_names=target_names, output_dict=True
    )

    # report dictionary -> DataFrame 
    # transpose, exclude final rows
    df_report = pd.DataFrame(report).transpose().iloc[:-3]

    fig, ax = plt.subplots(figsize=(10,6))

    df_report[['precision', 'recall', 'f1-score']].plot(
        kind='bar',
        ax=ax,
        rot=45,
        cmap=plt.cm.PuBuGn,
        edgecolor = 'black'
    )

    ax.set_title('Precision, Recall, F1', fontsize=14)
    ax.set_ylabel('Score', fontsize =12)
    ax.set_xlabel('Category', fontsize=12)
    ax.legend(loc='lower right')
    ax.grid(axis = 'y', linestyle = '--', alpha=0.7)


    filename = f'{prefix}per_class_metrics.png'
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)

    