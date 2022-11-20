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

    # todo delete
    # cm = [[403., 34., 230., 17., 30., 38., 19., 39., 154., 36.],
    #       [39., 617., 28., 13., 13., 22., 19., 19., 110., 120.],
    #       [45., 27., 408., 34., 189., 101., 107., 40., 39., 10.],
    #       [17., 21., 130., 155., 110., 309., 140., 42., 50., 26.],
    #       [23., 14., 154., 27., 494., 67., 101., 61., 47., 12.],
    #       [11., 11., 133., 78., 97., 499., 66., 57., 31., 17.],
    #       [6., 17., 95., 41., 143., 59., 560., 25., 39., 15.],
    #       [26., 18., 74., 32., 127., 121., 32., 524., 21., 25.],
    #       [106., 52., 86., 17., 17., 37., 4., 17., 615., 49.],
    #       [45., 185., 42., 34., 24., 29., 23., 53., 67., 498.]]
