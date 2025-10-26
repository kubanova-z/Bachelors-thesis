from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd

from src.plotting import plot_confusion_matrix, plot_metrics_bar_chart


def train_random_forest(X_train, y_train, X_test, y_test, target_names):
    print("\n" + "="*50)
    print("RANDOM FOREST")
    print("="*50)

    #inicilizacia a trenovanie modelu
    #n_estimators = pocet stromov
    #n_jobs = pouzitie dostupnych jadier procesora, aby bol trening rychlejsi
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    rf_model.fit(X_train, y_train)

    # predikcia na testovacich datach
    preds_labels = rf_model.predict(X_test)

    #evaluacia vysledkov
    report= classification_report(
        y_test,
        preds_labels,
        target_names=target_names,
        digits=4
    )
    print(report)

    #plotting statistik (labels -> int)
    classes = sorted(list(set(y_train)))
    class_to_idx = {cls: i for i, cls in enumerate (classes)}

    true_ids = y_test.map(class_to_idx).values
    preds_ids = pd.Series(preds_labels).map(class_to_idx).values

    plot_confusion_matrix(true_ids, preds_ids, target_names, prefix='rf_')
    plot_metrics_bar_chart(true_ids, preds_ids, target_names, prefix='rf_')

    return rf_model

