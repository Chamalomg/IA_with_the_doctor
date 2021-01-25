from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import seaborn as sns


def plot_loss(history):
    figure(figsize=(8, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def plot_confusion(conf):
    sns.heatmap(conf, square=True, annot=True, cbar=False
                , xticklabels=list(['Tyrion', 'Other'])
                , yticklabels=list(['Tyrion', 'Other']))
    plt.show()
