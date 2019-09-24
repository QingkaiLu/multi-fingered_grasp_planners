from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] == len(labels), "Different number of labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        #color = sqrt(x**2 + y**2)
        #plt.scatter(x, y, c=color, cmap='viridis')
        #plt.colorbar()
        plt.scatter(x, y) 
        label_str = "%.2f" % label
        plt.annotate(label_str,
                     xy=(x, y),
                     xytext=(2, 2),
                     textcoords='offset pixels',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)
    plt.clf()
    plt.close()


def plot_with_color(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] == len(labels), "Different number of labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    plt.scatter(low_dim_embs[:, 0], low_dim_embs[:, 1], c=labels, cmap='jet')
    plt.colorbar()
    plt.savefig(filename)
    plt.clf()
    plt.close()

def tsne_vis(data, labels, filename, color=False):
    #Process binary classification labels
    if len(labels.shape) == 2:
        #labels = np.argmax(labels, axis=1)
        print "Wrong labes dimension for tsne_vis!"
        return
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    low_dim_embs = tsne.fit_transform(data)
    #print 'low_dim_embs.shape', low_dim_embs.shape
    #print 'len(labels)', len(labels)
    if color:
        plot_with_color(low_dim_embs, labels, filename)
    else:
        plot_with_labels(low_dim_embs, labels, filename)

