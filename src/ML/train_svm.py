from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score, make_scorer
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np

def train_SVM(X_train, y_train, X_test, y_test, target_names, boost_factor = 1.0):
    print("\n" + "="*50)
    print("SUPPORT VECTOR MACHINE (SVM)")
    print("="*50)

    print_SVM_input(X_train, y_train)

    # vypocet vah pre triedy
    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes = classes,
        y = y_train
    )

    
    class_weights[1] *= boost_factor

    class_weight_dict = {cls: w for cls, w in zip(classes, class_weights)}

    print("\nComputed class weights:")
    for cls, w in class_weight_dict.items():
        print(f"  {cls}: {w:.4f}")


    # hladanie optimalneho C
    best_C = find_best_C(X_train, y_train, class_weight_dict)
    print(f"\nBest C found via StratifiedKFold: {best_C}")

    #inicializacia a trenovanie modelu
    #C relularizacny parameter (vyssie -> riziko overfittingu, nizsie -> viac chyb, ale vacsi margin)
    svm_model = LinearSVC(C=best_C,
                          class_weight=class_weight_dict,
                           random_state=42, 
                           verbose= 0)

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
    
    from src.plotting import plot_confusion_matrix, plot_metrics_bar_chart
    
    plot_confusion_matrix(true_ids, preds_ids, target_names, prefix='svm_')
    plot_metrics_bar_chart(true_ids, preds_ids, target_names, prefix='svm_')

    return svm_model


def print_SVM_input(X_train, y_train):
    print("\n--- SVM Model Input Data Summary ---")
    
    # Pre SciPy maticu používame X_train.shape
    print(f"X_train Shape (samples, features): {X_train.shape}")
    print(f"X_train Data Type: {X_train.dtype}")
    
    # Pre Pandas Series (y_train) používame .shape
    print(f"y_train Shape (labels): {y_train.shape}")
    print(f"y_train Data Type: {y_train.dtype}")
    
    print(f"Feature Vector Length (input_dim): {X_train.shape[1]}")
    print(f"Matrix Format: {type(X_train)}")
    print("-" * 30)
    
  
    
    if hasattr(X_train[0], "toarray"):  
        # TF-IDF sparse vector
        sample_vector = X_train[0].toarray()[0]
        print("First sample (first 10 TF-IDF feature values):")
    else:
        # Dense embedding (NumPy array)
        sample_vector = X_train[0]
        print("First sample (first 10 embedding values):")

    
    # Vypíšeme len prvých 10 prvkov z hustého poľa (array)
    print(sample_vector[:10]) 
    print("-" * 30)



def find_best_C(X_train, y_train, class_weight_dict):
    param_grid = {
        'C': [0.01, 0.1, 0.5, 1,2,5,10]
    }

    svm = LinearSVC(
        class_weight=class_weight_dict,
        random_state=42,
        max_iter=5000
    )

    scorer = make_scorer(f1_score, average ='weighted')  #predtym macro
    cv = StratifiedKFold(n_splits=5, shuffle = True, random_state=42)

    grid = GridSearchCV(
        svm,
        param_grid,
        scoring = scorer,
        cv = cv,
        n_jobs = -1,
        verbose = 0
    )

    grid.fit(X_train, y_train)
    return grid.best_params_['C']