from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import pandas as pd

def train_SVM(X_train, y_train, X_test, y_test, target_names):
    print("\n" + "="*50)
    print("SUPPORT VECTOR MACHINE (SVM)")
    print("="*50)

    print_SVM_input(X_train, y_train)

    #initialization of SVM model
    #C regulatory parameter (higher -> risk of overfitting, lower -> more errors, but larger margin)
    svm_model = LinearSVC(C=2.0, random_state=42, verbose= 0)

    
    svm_model.fit(X_train, y_train)

    # test data predictions
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
    
    from src.plotting import plot_confusion_matrix, plot_metrics_bar_chart
    
    plot_confusion_matrix(true_ids, preds_ids, target_names, prefix='svm_')
    plot_metrics_bar_chart(true_ids, preds_ids, target_names, prefix='svm_')

    return svm_model


def print_SVM_input(X_train, y_train):
    print("\n--- SVM Model Input Data Summary ---")
    
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

    
    print(sample_vector[:10]) 
    print("-" * 30)