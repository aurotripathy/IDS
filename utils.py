import matplotlib.pyplot as plt


def plot_graphs(history, component):
    """ component is either accuracy or loss """
    plt.title('Train ' + ' and Validation ' + component)
    plt.plot(history.history[component])
    plt.plot(history.history['val_'+component])
    plt.xlabel('Epochs')
    plt.ylabel(component)
    plt.legend([component, 'val_'+component])
    plt.grid()
    plt.show()
