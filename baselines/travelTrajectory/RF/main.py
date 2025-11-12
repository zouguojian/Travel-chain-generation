import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

def compute_macro_metrics(y_true, y_pred):
    """
    Compute macro-averaged Precision, Recall, and F1 score for multi-class classification.

    Parameters:
    y_true (list or np.array): True labels.
    y_pred (list or np.array): Predicted labels.

    Returns:
    tuple: (macro_precision, macro_recall, macro_f1)
    """
    # Get unique labels
    labels = np.unique(np.concatenate((y_true, y_pred)))
    n_classes = len(labels)

    # Create label to index mapping
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    # Initialize confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)

    # Populate confusion matrix
    for true, pred in zip(y_true, y_pred):
        cm[label_to_idx[true], label_to_idx[pred]] += 1

    # Compute per-class metrics
    precisions = []
    recalls = []
    f1s = []

    for i in range(n_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp

        # Precision for class i
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        precisions.append(precision)

        # Recall for class i
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recalls.append(recall)

        # F1 for class i
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)

    # Macro averages
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(f1s)

    return macro_precision, macro_recall, macro_f1

def load_dataset(dataset_dir):
    X = []
    Y = []
    # 加载数据到data字典中
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'), allow_pickle=True)
        X.append(cat_data['x'][:, :-2])
        Y.append(cat_data['x'][:, 5])

    X = np.concatenate(X)
    Y = np.array(np.concatenate(Y), dtype=int)
    return X, Y

X, Y = load_dataset(r'./data/')
print(X.shape)
print(Y.shape)


# 数据拆分：60%训练，20%验证，20%测试
X_train, X_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.3, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(2/3), shuffle=False)

# 仅对训练集进行额外shuffle（验证和测试集不shuffle）
train_indices = np.random.permutation(X_train.shape[0])
X_train = X_train[train_indices]
y_train = y_train[train_indices]

print(X_train.shape, y_train.shape)
# 初始化并训练随机森林分类器
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 验证集评估
y_pred_val = rf.predict(X_val)
val_accuracy = accuracy_score(y_val, y_pred_val)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# 测试集评估
y_pred_test = rf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f"Test Accuracy: {test_accuracy:.4f}")
macro_precision, macro_recall, macro_f1 = compute_macro_metrics(np.reshape(y_test, [-1]),
                                                                np.reshape(y_pred_test, [-1]))
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")
print(f"Macro F1: {macro_f1:.4f}")


# 特征提取：获取特征重要性
feature_names = ['City', 'License Plate', 'Vehicle Model', 'Day of Week', 'Minute of Day']
importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# 打印特征重要性
print("\nFeature Importances:")
print(feature_importance_df)

# 可选：使用前k个重要特征重新训练（例如前5个）
k = 5
top_features = feature_importance_df.head(k)['Feature'].values
top_indices = [feature_names.index(f) for f in top_features]
X_train_selected = X_train[:, top_indices]
X_val_selected = X_val[:, top_indices]
X_test_selected = X_test[:, top_indices]

rf_selected = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selected.fit(X_train_selected, y_train)

# 重新评估
y_pred_val_selected = rf_selected.predict(X_val_selected)
val_accuracy_selected = accuracy_score(y_val, y_pred_val_selected)
print(f"\nValidation Accuracy with Top {k} Features: {val_accuracy_selected:.4f}")

y_pred_test_selected = rf_selected.predict(X_test_selected)
test_accuracy_selected = accuracy_score(y_test, y_pred_test_selected)
print(f"Test Accuracy with Top {k} Features: {test_accuracy_selected:.4f}")

macro_precision, macro_recall, macro_f1 = compute_macro_metrics(np.reshape(y_test, [-1]),
                                                                np.reshape(y_pred_test_selected, [-1]))
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")
print(f"Macro F1: {macro_f1:.4f}")