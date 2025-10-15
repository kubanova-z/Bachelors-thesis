from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import pandas as pd

def train_SVM(X_train, y_train, X_test, y_test, target_names):
    print("\n" + "="*50)
    print("SUPPORT VECTOR MACHINE (SVM)")
    print("="*50)

    #inicializacia a trenovanie modelu
    #C relularizacny parameter (vyssie -> riziko overfittingu, nizsie -> viac chyb, ale vacsi margin)
    svm_model = LinearSVC(C=2.0, random_state=42, verbose= 0)

    #
    svm_model.fit(X_train, y_train)

    # predikcia na testovacich datach
    # labels sa spracovavaju priamo ako textove retazce, netreba ich konvertovat na cisla ako pri NN
    preds_labels = svm_model.predict(X_test)

    #evaluation

    #classification report (porovnanie skutocnych a predikovanych labels)
    print("\n--- SVM CLASSIFICATION REPORT ---")
    report = classification_report(
        y_test, 
        preds_labels, 
        target_names=target_names, 
        digits=4
    )
    print(report)

    #confusion matrix
    #matrics bar chart

    classes = sorted(list(set(y_train)))
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    
    true_ids = y_test.map(class_to_idx).values
    preds_ids = pd.Series(preds_labels).map(class_to_idx).values
    
    from .plotting import plot_confusion_matrix, plot_metrics_bar_chart
    
    plot_confusion_matrix(true_ids, preds_ids, target_names, prefix='svm_')
    plot_metrics_bar_chart(true_ids, preds_ids, target_names, prefix='svm_')

    return svm_model