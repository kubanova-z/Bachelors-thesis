from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score, make_scorer, recall_score
from sklearn.utils.class_weight import compute_class_weight
from src.plotting import plot_confusion_matrix, plot_metrics_bar_chart
import pandas as pd
import numpy as np

def train_SVM(X_train, y_train, X_test, y_test, target_names, boost_factor = 1.0):
    print("\n" + "="*50)
    print("SUPPORT VECTOR MACHINE (SVM)")
    print("="*50)

    print_SVM_input(X_train, y_train)

    # compute class weights
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


    # optimal value of C paramter - maximizing F1 score or recall for minority class
    #best_C = find_best_C(X_train, y_train, class_weight_dict)
    best_C = find_best_C_recall_1(X_train, y_train, class_weight_dict)
    print(f"\nBest C found via StratifiedKFold: {best_C}")

    #initialization and training of the model
    #C regularization parameter - higher C - small margin, larger emphasis on correct classification, but risk of overfitting
    svm_model = LinearSVC(C=best_C,
                          class_weight=class_weight_dict,
                           random_state=42, 
                           verbose= 0)

    #
    svm_model.fit(X_train, y_train)


    preds_labels = svm_model.predict(X_test)

    #evaluation
    print("\n--- SVM CLASSIFICATION REPORT ---")
    report = classification_report(
        y_test, 
        preds_labels, 
        target_names=target_names, 
        digits=4
    )
    print(report)

    classes = sorted(list(set(y_train)))
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    
    true_ids = y_test.map(class_to_idx).values
    preds_ids = pd.Series(preds_labels).map(class_to_idx).values
    
   
    
    plot_confusion_matrix(true_ids, preds_ids, target_names, prefix='svm_')
    plot_metrics_bar_chart(true_ids, preds_ids, target_names, prefix='svm_')

    return svm_model, preds_ids, true_ids


def print_SVM_input(X_train, y_train):
    print("\n--- SVM Model Input Data Summary ---")
    

    print(f"X_train Shape (samples, features): {X_train.shape}")
    print(f"X_train Data Type: {X_train.dtype}")
    

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

    

    print(sample_vector[:10]) 
    print("-" * 30)


# C - regularizacny parameter
# velke C - maly margin, vacsi doraz na spravnu klasifikaciu, ale riziko pretrenovania
# male C - vacsi margin, lepsia generalizacia, ale viac nespravnych klasifikacii, riziko underfittingu

# testing different values of C parameter - maximizing F1 score or recall for minority class
def find_best_C(X_train, y_train, class_weight_dict):
    param_grid = {
        'C': [0.01, 0.1, 0.5, 1,2,5,10]
    }

    svm = LinearSVC(
        class_weight=class_weight_dict,
        random_state=42,
        max_iter=5000
    )

    scorer = make_scorer(f1_score, average ='weighted')  
    cv = StratifiedKFold(n_splits=5, shuffle = True, random_state=42)   # 5 fold stratified cross validation
    # train data split into 5 equal parts (folds) - model is trained and evaluated 5 times, each time using a different fold as the validation set and the remaining 4 folds as the training set. StratifiedKFold ensures that the class distribution is preserved in each fold, which is important for imbalanced datasets.
    # stratified - folds preserves class ratio of dataset

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


def find_best_C_recall_1(X_train, y_train, class_weight_dict):
    param_grid = {
        'C': [0.01, 0.1, 0.5, 1,2,5,10]
    }

    svm = LinearSVC(
        class_weight=class_weight_dict,
        random_state=42,
        max_iter=5000
    )

    scorer = make_scorer(recall_score, pos_label=1) # maximizing recall for minority class
    cv = StratifiedKFold(n_splits=5, shuffle = True, random_state=42)   # 5 fold stratified cross validation
    # train data split into 5 equal parts (folds) - model is trained and evaluated 5 times, each time using a different fold as the validation set and the remaining 4 folds as the training set. StratifiedKFold ensures that the class distribution is preserved in each fold, which is important for imbalanced datasets.
    # stratified - folds preserves class ratio of dataset

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

