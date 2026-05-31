from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.font_manager as fm
import os
os.makedirs('/img', exist_ok=True)

# Path to font file
font_path ="/home/xkubanova_126831/bakalarka/font/Open_Sans/OpenSans-VariableFont_wdth,wght.ttf"
fm.fontManager.addfont(font_path)
# Set as default font
plt.rcParams['font.family'] = 'Open Sans'

#CONFUSION MATRIX
def plot_confusion_matrix(true_ids, preds_ids, target_names, prefix=''):
    cm = confusion_matrix(true_ids, preds_ids)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_normalized,
        display_labels=target_names
    )

    custom_cmap = LinearSegmentedColormap.from_list(
        "thesis_blue",
        ["#ffffff", "#abd2fa", "#2f52e0"]
    )

    fig, ax = plt.subplots(figsize=(10, 10))

    disp.plot(
        cmap=custom_cmap,
        ax=ax,
        xticks_rotation=45,
        values_format='.1%'
    )


    ax.set_title('Confusion Matrix', fontsize=20, fontweight='bold')
    ax.set_ylabel('True Category', fontsize=16, fontweight='bold')
    ax.set_xlabel('Predicted Category', fontsize=16, fontweight='bold')

    ax.tick_params(axis='both', labelsize=14)

    for text in disp.text_.ravel():
        text.set_fontsize(30)

    plt.tight_layout()

    filename = f'/img/{prefix}confusion_matrix_results.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close(fig)    
    
    


#LEARNING CURVE
def plot_learning_curve(epochs, train_loss, test_acc, prefix=''):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
    epoch_range = range(1, epochs+1)

    ax1.plot(
    epoch_range,
    train_loss,
    label='Training loss',
    color='#2f52e0',
    linewidth=2
    )
    ax1.set_title('Training loss per epoch', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (Cross Entropy)', fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(
        epoch_range,
        test_acc,
        label='Test accuracy',
        color='#a5835a',
        linewidth=2
    )
    ax2.plot(epoch_range, test_acc, label='Test accuracy', color='blue')
    ax2.set_title('Test accuracy per Epoch', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    filename = f'/img/{prefix}learning_curve.png'

    plt.show()
    plt.savefig(filename)
    plt.tight_layout() 
    plt.close(fig)
    

    
#BAR CHART
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

    df_report = df_report[['precision', 'recall', 'f1-score']].round(3)

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

    filename = f'/img/{prefix}per_class_metrics.png'
    plt.savefig(filename, dpi=300)

    plt.show()
    plt.close(fig)

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

    filename_table = f'/img/{prefix}per_class_metrics_table.png'

    plt.savefig(
        filename_table,
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()
    plt.close(fig)


    df_report.to_csv(
        f'/img/{prefix}per_class_metrics.csv'
    )

    return df_report



#ROC CURVE

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


PRIMARY = '#2f52e0'
NEUTRAL = '#7f7979'


def plot_roc_curve(labels, probs, prefix=''):

    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(
        fpr,
        tpr,
        color=PRIMARY,
        lw=2,
        label=f'ROC curve (AUC = {roc_auc:.3f})'
    )

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

    filename = f'/img/{prefix}roc_curve.png'

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()
    plt.close(fig)

    return roc_auc


#PR CURVE
from sklearn.metrics import precision_recall_curve, average_precision_score


def plot_precision_recall_curve(labels, probs, prefix=''):

    precision, recall, _ = precision_recall_curve(labels, probs)
    pr_auc = average_precision_score(labels, probs)

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(
        recall,
        precision,
        color=PRIMARY,
        lw=2,
        label=f'PR curve (AP = {pr_auc:.3f})'
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

    filename = f'/img/{prefix}precision_recall_curve.png'

    plt.savefig(
       filename,
      dpi=300,
       bbox_inches='tight'
    )

    plt.show()
    plt.close(fig)

    return pr_auc