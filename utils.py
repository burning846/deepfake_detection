import numpy as np
from sklearn.metrics import roc_curve, auc

def cal_tar_far(y_scores, y_true):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # 打印结果
    print(f"False Positive Rate: {fpr}")
    print(f"True Positive Rate: {tpr}")
    print(f"AUC: {roc_auc}")
