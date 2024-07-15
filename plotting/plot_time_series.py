import matplotlib.pyplot as plt
import numpy as np

class TimeSeriesData:
    def __init__(self, data, index=None, labels=None):
        """
        Initialize the TimeSeriesData object.

        Parameters:
        - data (np.ndarray): array of time series data
          - if a 2D array is given, each array is a separate time series
        - index (np.ndarray): range of values indexing time series data
          - must have same shape as data
        - labels: array of strings for plotting legend (must have same 1st dimension as data
        """
        self.data = data

        if index is None:
            self.index = np.ones_like(data) * np.arange(data.shape[-1])
        else:
            self.index = index

        self.labels = labels


    def plot_time_series(self, xlabel, ylabel):
        print("data shape: ", self.data.shape)
        print("index shape: ", self.index.shape)

        plt.style.use('ggplot')
        plt.rcParams["figure.figsize"] = (19,11)
        plt.rcParams["font.size"] = 36
        plt.rcParams["xtick.color"] = 'black'
        plt.rcParams["ytick.color"] = 'black'
        plt.rcParams["axes.edgecolor"] = 'black'
        plt.rcParams["axes.linewidth"] = 1  

        colors = [
            '#1f77b4',  # blue
            '#ff7f0e',  # orange
            '#2ca02c',  # green
            '#d62728',  # red
            '#9467bd',  # purple
            '#8c564b',  # brown
            '#e377c2',  # pink
            '#7f7f7f'   # gray
        ]
        dashes = [[1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0]]
        markers = ['o','d','s','*','P','H','<','8','X','v','o']

        plt.figure()

        for i in range(len(self.data)):
            print(i)
            plt.plot(self.index[i], self.data[i], label=self.labels[i], color=colors[i], dashes=dashes[i], linewidth=6, alpha=1.0) #, marker=MARKERS[i], markersize=25)
        
        plt.xlabel(xlabel, color='black')
        plt.ylabel(ylabel, color='black')
        plt.grid(color='grey', linestyle='--', linewidth=.5)
        # plt.xticks
        # plt.yticks
        # plt.xlim
        # plt.ylim
        # plt.title
        if len(data.shape) == 2 and data.shape[0] > 1:
            plt.legend(fancybox=True,handlelength=1,shadow=False,loc='upper left',ncol=1,fontsize=33, framealpha=1.0, edgecolor='black', borderpad=0.5, borderaxespad=1.0)

        plt.show()

if __name__ == "__main__":
    data = np.random.rand(2, 20)
    labels = ['series1', 'series2']
    time_series = TimeSeriesData(data, labels=labels)
    time_series.plot_time_series("time", "unit")
