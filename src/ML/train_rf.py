from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd

from src.plotting import plot_confusion_matrix, plot_metrics_bar_chart


def train_random_forest(X_train, y_train, X_test, y_test, target_names):
    print("\n" + "="*50)
    print("RANDOM FOREST")
    print("="*50)

    #print_rf_input(X_train, y_train)

    #initialization of Random Forest model
    #n_estimators = number of decision trees
    #n_jobs = utilization of available processor cores to make training faster
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    rf_model.fit(X_train, y_train)

    # test data predictions
    preds_labels = rf_model.predict(X_test)

    # evaluation
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

    return rf_model



def print_rf_input(X_train, y_train):
    print("\n--- RF Model Input Data Summary ---")
    
    # SciPy matrix
    print(f"X_train Shape (samples, features): {X_train.shape}")
    print(f"X_train Data Type: {X_train.dtype}")
    
    # Pandas Series 
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
    
    print("Prvá vzorka (prvých 10 hodnôt):")
    # Vypíšeme len prvých 10 prvkov z hustého poľa (array)
    print(sample_vector[:10]) 
    print("-" * 30)