def plot_barchart(df_corect, df_wrong):
    df = pd.concat([df_corect, df_wrong])
    df = pd.DataFrame(df.mean(axis=0), columns=['Certainty'])
    df.reset_index(inplace=True)
    df = df.rename(columns={'index': 'predictions'})
    df.sort_values('predictions').plot.bar(x="predictions", y="Certainty", rot=70, figsize=(10, 5))
    plt.show(block=True)


def plot_histogram(df):
    histogram = df[sorted(df.columns)]
    axarr = histogram.hist(sharex=True, sharey=True, figsize=(20, 10))

    for ax in axarr.flatten():
        ax.set_xlabel("Certainty")
        ax.set_ylabel("Count")
    plt.show(block=True)


class Charts:

    def __init__(self, certainties, predictions, labels) -> None:
        super(Charts, self).__init__()
        self.certainties = certainties
        self.predictions = predictions
        self.labels = labels

        self.df_pred_correct = pd.DataFrame()
        self.df_pred_wrong = pd.DataFrame()
        self.df_pred_correct_after = pd.DataFrame()
        self.df_pred_wrong_after = pd.DataFrame()

        self.build_dataframes()

    def build_dataframes(self):
        for i in range(0, self.certainties.size):
            if self.predictions[i] == self.labels[i]:
                temp_df = pd.DataFrame([[self.certainties[i]]], columns=["c " + str(self.labels[i])])
                self.df_pred_correct = pd.concat([self.df_pred_correct, temp_df])
                temp_df = pd.DataFrame([[self.certainties[i]]], columns=[str(self.labels[i]) + " c"])
                self.df_pred_correct_after = pd.concat([self.df_pred_correct_after, temp_df])
            else:
                temp_df = pd.DataFrame([[self.certainties[i]]], columns=["w " + str(self.predictions[i])])
                self.df_pred_wrong = pd.concat([self.df_pred_wrong, temp_df])
                temp_df = pd.DataFrame([[self.certainties[i]]], columns=[str(self.labels[i]) + " w"])
                self.df_pred_wrong_after = pd.concat([self.df_pred_wrong_after, temp_df])

    def plot_barcharts(self):
        print('\nAverage certainty per class - correct on the left and wrong on the right')
        plot_barchart(self.df_pred_correct, self.df_pred_wrong)
        print('\nAverage certainty for comparing differences between each class correct and wrong')
        plot_barchart(self.df_pred_correct_after, self.df_pred_wrong_after)

    def plot_histograms(self):
        print('\nHistogram correct prediction certainty per class')
        plot_histogram(self.df_pred_correct)
        print('\nHistogram wrong prediction certainty per class')
        plot_histogram(self.df_pred_wrong)