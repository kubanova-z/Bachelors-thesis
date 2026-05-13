import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap

# Path to your font file
font_path ="/home/xkubanova_126831/bakalarka/font/Open_Sans/OpenSans-VariableFont_wdth,wght.ttf"

# Register font
fm.fontManager.addfont(font_path)

# Set as default font
plt.rcParams['font.family'] = 'Open Sans'


#CONFUSION MATRIX
def plot_confusion_matrix(true_ids, preds_ids, target_names, prefix=''):
    cm = confusion_matrix(true_ids, preds_ids)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    #display object
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_normalized, 
        display_labels=target_names
    )

    custom_cmap = LinearSegmentedColormap.from_list(
        "thesis_blue",
        ["#ffffff", "#abd2fa", "#2f52e0"]
    )

    #plot matrix
    fig, ax = plt.subplots(figsize=(10,10))
    disp.plot(
        cmap=custom_cmap,
        ax=ax,
        xticks_rotation=45,
        values_format='.1%'
    )

    ax.set_title('Confusion matrix', fontsize=20, fontweight='bold')
    ax.set_ylabel('True Category', fontsize=16, fontweight='bold')
    ax.set_xlabel('Predicted Category', fontsize=16, fontweight='bold')

    # Tick labels
    ax.tick_params(axis='both', labelsize=14)

    # Numbers inside cells
    for text in disp.text_.ravel():
        text.set_fontsize(30)

    plt.tight_layout()

    plt.show()
    filename = f'{prefix}confusion_matrix_results.png'
    plt.savefig(filename) 
    plt.close(fig) 
    
    
    


#LEARNING CURVE
# LEARNING CURVE
def plot_learning_curve(epochs, train_loss, val_loss, test_acc=None, prefix=''):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    epoch_range = range(1, epochs + 1)

    # Training + Validation loss
    ax1.plot(
        epoch_range,
        train_loss,
        label='Training loss',
        color='#2f52e0',
        linewidth=2
    )

    ax1.plot(
        epoch_range,
        val_loss,
        label='Validation loss',
        color='#a5835a',
        linewidth=2
    )

    ax1.set_title('Loss per epoch', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (Cross Entropy)', fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Test accuracy
    if test_acc is not None:
        ax2.plot(
            epoch_range,
            test_acc,
            label='Test accuracy',
            color='#a5835a',
            linewidth=2
        )

        ax2.set_title('Test accuracy per epoch', fontsize=14)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)

    filename = f'{prefix}learning_curve.png'

    plt.tight_layout()
    plt.savefig(filename)

    plt.show()
    plt.close(fig)

    

    
# bar chart - F1 score
def plot_metrics_bar_chart(true_ids, preds_ids, target_names, prefix=''):

    report = classification_report(
        true_ids,
        preds_ids,
        target_names=target_names,
        output_dict=True
    )

    df_report = (
        pd.DataFrame(report)
        .transpose()
        .iloc[:-3]
    )

    # Round for readability
    df_report = df_report[['precision', 'recall', 'f1-score']].round(3)

    # -----------------------
    # BAR CHART
    # -----------------------

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#2f52e0', '#abd2fa', '#a5835a']

    df_report.plot(
        kind='bar',
        ax=ax,
        rot=45,
        color=colors,
        edgecolor='black'
    )

    ax.set_title('Precision, Recall, F1', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('Category', fontsize=12)

    ax.legend(loc='lower right')

    ax.grid(
        axis='y',
        linestyle='--',
        alpha=0.7
    )

    plt.tight_layout()

    filename = f'{prefix}per_class_metrics.png'
    plt.savefig(filename, dpi=300)

    plt.show()
    plt.close(fig)

    # -----------------------
    # TABLE FIGURE
    # -----------------------

    fig, ax = plt.subplots(figsize=(8, 2 + len(df_report) * 0.5))

    ax.axis('off')

    table = ax.table(
        cellText=df_report.values,
        colLabels=df_report.columns,
        rowLabels=df_report.index,
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)

    ax.set_title(
        'Per-class Evaluation Metrics',
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    plt.tight_layout()

    filename_table = f'{prefix}per_class_metrics_table.png'

    plt.savefig(
        filename_table,
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()
    plt.close(fig)

    # -----------------------
    # OPTIONAL EXPORTS
    # -----------------------

    df_report.to_csv(
        f'{prefix}per_class_metrics.csv'
    )

    return df_report
    

# -----------------------
# ROC CURVE (MULTICLASS)
# -----------------------
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np


PRIMARY = '#2f52e0'
NEUTRAL = '#7f7979'


def plot_roc_curve(labels, probs, prefix=''):


    labels = np.array(labels)
    probs = np.array(probs)

    n_classes = probs.shape[1]

    # probs shape: [n_samples, n_classes]
    n_classes = probs.shape[1]

    # one-hot encoding of labels
    labels_bin = label_binarize(
        labels,
        classes=np.arange(n_classes)
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    roc_auc_scores = []

    # plot ROC for each class
    for i in range(n_classes):

        fpr, tpr, _ = roc_curve(
            labels_bin[:, i],
            probs[:, i]
        )

        roc_auc = auc(fpr, tpr)
        roc_auc_scores.append(roc_auc)

        ax.plot(
            fpr,
            tpr,
            lw=2,
            label=f'Class {i} (AUC = {roc_auc:.3f})'
        )

    # random classifier
    ax.plot(
        [0, 1],
        [0, 1],
        color=NEUTRAL,
        lw=2,
        linestyle='--',
        label='Random classifier'
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)

    ax.set_title(
        'Receiver Operating Characteristic (ROC)',
        fontsize=14,
        fontweight='bold'
    )

    ax.legend(loc='lower right')

    ax.grid(
        True,
        linestyle='--',
        alpha=0.4
    )

    plt.tight_layout()

    filename = f'{prefix}roc_curve.png'

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()
    plt.close(fig)

    return roc_auc_scores


# -----------------------
# PR CURVE (MULTICLASS)
# -----------------------

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score
)


def plot_precision_recall_curve(labels, probs, prefix=''):

    labels = np.array(labels)
    probs = np.array(probs)

    n_classes = probs.shape[1]

    n_classes = probs.shape[1]

    labels_bin = label_binarize(
        labels,
        classes=np.arange(n_classes)
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    pr_auc_scores = []

    for i in range(n_classes):

        precision, recall, _ = precision_recall_curve(
            labels_bin[:, i],
            probs[:, i]
        )

        pr_auc = average_precision_score(
            labels_bin[:, i],
            probs[:, i]
        )

        pr_auc_scores.append(pr_auc)

        ax.plot(
            recall,
            precision,
            lw=2,
            label=f'Class {i} (AP = {pr_auc:.3f})'
        )

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)

    ax.set_title(
        'Precision–Recall Curve',
        fontsize=14,
        fontweight='bold'
    )

    ax.legend(loc='lower left')

    ax.grid(
        True,
        linestyle='--',
        alpha=0.4
    )

    plt.tight_layout()

    filename = f'{prefix}precision_recall_curve.png'

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()
    plt.close(fig)

    return pr_auc_scores