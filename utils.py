from matplotlib import pyplot as plt

def plot_roc_curve(fpr, tpr):
    """Plot ROC curve given the false positive rate (fpr) and true positive
    """
    plt.plot([0, 1], [0, 1], 'k--')
    plt.show()