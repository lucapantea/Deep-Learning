import seaborn as sn
import pandas as pd
from matplotlib import pyplot as plt

def plot_cm(cm):
    # Plot confusion matrix
    df_cm = pd.DataFrame(cm, index=[i for i in "0123456789"],
                         columns=[i for i in "0123456789"])
    plt.figure(figsize=(10, 7))
    ax = sn.heatmap(df_cm, annot=True, fmt='g')
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    plt.yticks(rotation=0)
    plt.show()
