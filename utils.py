import matplotlib.pyplot as plt
import numpy as np

def mat_to_figure(data:np.ndarray,x,y):
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.matshow(data)
    ax.set_xticklabels(['']+x)
    ax.set_yticklabels(['']+y)
    for (i, j), z in np.ndenumerate(data):
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
    return fig

# def normalize(data:np.array,)