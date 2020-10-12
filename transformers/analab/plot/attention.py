import matplotlib.ticker as ticker

from .plot import Plot


class AttenMatPlot(Plot):
    def __init__(self, ax):
        self.ax = ax

    def __call__(self, data, x_labels, y_labels):
        cax = self.ax.matshow(data, cmap='bone')
        ax.tick_params(labelsize=15)
        ax.set_xticklabels(x_labels, rotation=45)
        ax.set_yticklabels(y_labels)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))


class AttenArrowPlot(Plot):
    def __init__(self, ax):
        self.ax = ax

    def __call__(self, data, x_labels, y_labels):
        width = 3
        word_height = 1
        words = ["[CLS]", "I", "like", "ice", "cream", "[SEP]"]

        self.ax.axis("off")
        for position, word in enumerate(data):
            self.ax.text(0, 2. - float(position) * word_height, word, ha="right", va="center")        
            self.ax.text(width, 2. - float(position) * word_height, word, ha="left", va="center")        
