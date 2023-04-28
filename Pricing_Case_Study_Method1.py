"""
Method1
"""

# Packages imports
from scipy.stats import poisson
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.stats.api as sms
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
from scipy.stats import ttest_ind
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score, recall_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")





class Pricing_Case_Study:

    def __init__(self, months_of_service, rider_charge, total_riders, max_monthly_rides, lambda_):
        self.months_of_service = months_of_service
        self.rider_charge = rider_charge
        self.total_riders = total_riders
        self.max_monthly_rides = max_monthly_rides
        self.lambda_ = lambda_

    def plot_drivers_histogram(self, df_accepted, df_rejected):
        fig, ax = plt.subplots(2, 1, figsize=(9, 12))
        ax[0].hist(df_accepted, bins=50)
        ax[0].set_ylabel('Accepted drivers')
        ax[0].axvline(df_accepted.mean(), color='magenta', linestyle='dashed', linewidth=2)
        ax[1].hist(df_rejected, bins=50)
        ax[1].set_ylabel('Rejected drivers')
        ax[1].axvline(df_rejected.mean(), color='magenta', linestyle='dashed', linewidth=2)
        ax[1].set_xlabel('Payment ($)')
        fig.suptitle('Payment Distribution')
        return plt.show()

    def search_space_plot(self):
        x_ = np.linspace(1, 12, 12)
        y_ = np.linspace(1, 10, 10)
        z_ = np.linspace(20, 40, 20)
        x, y, z = np.meshgrid(x_, y_, z_)
        fig = plt.figure(figsize=(6, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z)
        ax.set_xlabel('Month')
        ax.set_ylabel('Ride')
        ax.set_zlabel('Pay')
        ax.set_title('Pricing Grid Search Space')
        return plt.show()

    def descriptive_analysis(self):
        print("Amount of Nan values :",df.isna().sum()) # no Nan values
        print(df['ACCEPTED'].value_counts(normalize=True)) # 527 accepted vs 473 rejected: almost balanced
        df_accepted = df.loc[df['ACCEPTED'] == 1, 'PAY'].sort_values().reset_index(drop=True)
        df_rejected = df.loc[df['ACCEPTED'] == 0, 'PAY'].sort_values().reset_index(drop=True)

        #plots
        sns.boxplot(data=df, x='ACCEPTED', y='PAY')
        plt.title("Boxplot")
        obj.plot_drivers_histogram(df_accepted, df_rejected) # plot payment distribution of accepted vs rejected drivers
        obj.search_space_plot()
        plt.show()

        # statistical significance
        stat, p_value = ttest_ind(df_accepted.values, df_rejected.values)
        print(f"t-test: statistic={stat:.4f}, p-value={p_value:.4f}")

    def pois_dist_probs(self, ROI, MMR):
        """ Use poisson_distribution where
            ROI : Ride Occurance Index (k in poisson distribution)
            MMR : Matched Monthly Ride (mu or lambda in poisson distribution)
            returns
            PMF: Probability Mass Function, CDF: Cumulative Distribution Function
        """
        PMF = poisson.pmf(k=ROI, mu=MMR)
        CDF = poisson.cdf(k=ROI, mu=MMR)
        return PMF, CDF

    def get_driver_accept_prob(self, df):
        # logistic regression
        X, y = df['PAY'].values.reshape(-1, 1), df['ACCEPTED'].values.reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
        model = LogisticRegression().fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        acc = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        print("MSE: ", mse,'\n', "ACC: ", acc,'\n',"precision: ", precision,'\n',"recall: ", recall)
        return model


    def driver_accept_prob(self, model, y):
        y = np.expand_dims(np.array([y]), 0)
        return model.predict_proba(y)[0][1]


if __name__ == "__main__":
    # attributes
    months_of_service = 12
    rider_charge = 30
    total_riders = 10000
    max_monthly_rides = 1000
    lambda_1stMonth = 1
    # create an object with above attributes
    obj = Pricing_Case_Study(months_of_service, rider_charge, total_riders, max_monthly_rides, lambda_1stMonth)

    # get driverAcceptanceData
    df = pd.read_csv('driverAcceptanceData.csv').drop('Unnamed: 0',axis=1)

    # # Descriptive studies
    obj.descriptive_analysis()

    # Train similar drivers data to get driver's acceptance probability
    model = obj.get_driver_accept_prob(df)

    #
    # # Follow one rider in the 1st month with $30 payment for each ride
    data = [] # To store grids info (Profit, Month, Pay, ROI, MMRC) in a list to be added later to a dataframe
    MMRP = {} # To store MMRCs to determine MMR for the next month
    """
        Write grid data for Profit, Month, Pay, ROI, MMRP in an array:
        data.append([Profit, Month, Pay, ROI, MMRP]) 
        and then transfer it to a data frame:
        grid = pd.DataFrame(data, columns=['Profit', 'Month', 'Pay', 'ROI', 'MMRC'])
    """

    Profit = 0
    Month = 1
    ROI = 1 # Ride Occurance Index (k in Poisson distribution)
    MMR = 1 # Matched Monthly Ride (mu or lambda in Poisson distribution)
    MMRC = 0 # Matched Monthly Ride Counter

    for Month in range(1,13):
        for Pay in range(20,41):

            # Get Driver's Accept Probability (DAP)
            DAP = obj.driver_accept_prob(model, Pay)
            # print("DAP :",DAP)
            if DAP >= 0.5:
                PMF = 0
                CDF = 0
                while CDF <= 0.99:
                    MMRC += 1
                    # Get Riders Request Probability
                    PMF, CDF = obj.pois_dist_probs(ROI, MMR) # PMF: Probability Mass Function, CDF: Cumulative Distribution Function
                    # print("PMF :", PMF, "CDF :", CDF)

                    # Calculate the probability-based profit
                    if Pay<30:
                        Profit += (30-Pay) * PMF
                    elif Pay==30:
                        Profit += 0.5 * PMF
                    else:
                        Profit += (1/(2**(np.abs(30-Pay)+1))) * PMF # convert range of negative values to a range from 0 to 1
                    # print("Profit :",Profit)
                    ROI += 1
            # Record Grid Information(Profit, Month, Pay, ROI, MMRC)
            data.append([Profit, Month, Pay, ROI, MMRC])
            MMRP[(Month,Pay)] = MMRC
            Pay += 1
            ROI = 1
            Profit = 0
            MMRC = 0
            # print(data)
            # print(MMRP)

    grid = pd.DataFrame(data, columns=['Profit', 'Month', 'Pay', 'ROI', 'MMRC'])

    print(grid)



