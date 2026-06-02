from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, make_scorer, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from imblearn.ensemble import BalancedRandomForestClassifier
import pandas as pd
import numpy as np

from src.plotting import plot_confusion_matrix, plot_metrics_bar_chart


def train_random_forest(X_train, y_train, X_test, y_test, target_names, boost_factor = 1.0):
    print("\n" + "="*50)
    print("RANDOM FOREST")
    print("="*50)

    #print_rf_input(X_train, y_train)

    classes = np.array(sorted(list(set(y_train))))
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )
    
    print("Original class weights:", class_weights)

 
    class_weights[1] *= boost_factor

    print("Boosted class weights:", class_weights)

    class_weight_dict = {cls: w for cls, w in zip(classes, class_weights)}

    #initialization of the model
    #n_estimators = number of decision trees


    #GridSearchCV - hyperparameter tuning 
    param_grid = {
        'n_estimators': [50, 100, 200], # number of trees in the forest
        'max_depth': [None, 10, 20, 30], # max depth of each tree
        'min_samples_split': [2, 5, 10], # minimum number of samples required to split an internal node (lower - more splits, risk of overfitting, higher - fewer splits, more bias)
        'min_samples_leaf': [1, 2, 4] # minimum number of samples required to be at a leaf node (lower - more splits, risk of overfitting, higher - fewer splits, more bias)
    }



    rf_model = BalancedRandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1, 
    class_weight=class_weight_dict,  
    verbose=0
)

    # Maximiying F1 score or recall for the minority class
    #scorer = make_scorer(f1_score, average='macro')
    scorer = make_scorer(recall_score, pos_label=1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(rf_model, param_grid, scoring=scorer, cv=cv, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    best_rf = grid.best_estimator_
    print(f"\nBest Random Forest parameters found via StratifiedKFold: {grid.best_params_}")



    preds_labels = best_rf.predict(X_test)

    report= classification_report(
        y_test,
        preds_labels,
        target_names=target_names,
        digits=4
    )
    print(report)


    classes = sorted(list(set(y_train)))
    class_to_idx = {cls: i for i, cls in enumerate (classes)}

    true_ids = y_test.map(class_to_idx).values
    preds_ids = pd.Series(preds_labels).map(class_to_idx).values

    plot_confusion_matrix(true_ids, preds_ids, target_names, prefix='rf_')
    plot_metrics_bar_chart(true_ids, preds_ids, target_names, prefix='rf_')

    return best_rf, preds_ids, true_ids



def print_rf_input(X_train, y_train):
    print("\n--- RF Model Input Data Summary ---")
    
   
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
    
    print("First sample (first 10 values):")
    print(sample_vector[:10]) 
    print("-" * 30)



