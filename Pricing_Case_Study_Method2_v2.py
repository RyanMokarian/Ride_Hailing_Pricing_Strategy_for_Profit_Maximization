"""
Method2
v1:
Changed Metric to reflect loss (negative profit)
Calculated Profit for the 1st month over 41 level of payments
v2:
Only driver's rides for 23, 30, and 37 driver's payment are considered.
The search space is reduced from 41pays X 12months (22563490300366186081) to 3pays x 12months (531,441) different combinations.
After getting result for ROI=1, the process can be repeated for ROI = 2, 3, ...
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

    def __init__(self, months_of_service, rider_charge, total_riders, rides_monthly_cap, lambda_, min_pay, max_pay, pay_increment):
        self.months_of_service = months_of_service
        self.rider_charge = rider_charge
        self.total_riders = total_riders
        self.rides_monthly_cap = rides_monthly_cap
        self.lambda_ = lambda_
        self.min_pay = min_pay
        self.max_pay = max_pay
        self.pay_increment = pay_increment

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

        # #plots
        # sns.boxplot(data=df, x='ACCEPTED', y='PAY')
        # plt.title("Boxplot")
        # obj.plot_drivers_histogram(df_accepted, df_rejected) # plot payment distribution of accepted vs rejected drivers
        # obj.search_space_plot()
        # plt.show()

        # # statistical significance
        # stat, p_value = ttest_ind(df_accepted.values, df_rejected.values)
        # print(f"t-test: statistic={stat:.4f}, p-value={p_value:.4f}")

    def pois_dist_probs(self, MMR, max_rides_num):
        """ Use poisson_distribution where
            ROI : Ride Occurance Index (k in poisson distribution)
            MMR : Matched Monthly Ride (mu or lambda in poisson distribution)
            returns
            ROI_RideNum: a list of paired "ROI" and number of rides (Probability Mass Function * rides_monthly_cap)
        """
        PMF = 1
        ROI = 1
        ROI_RideNum = []
        while PMF>0.001:
            PMF = poisson.pmf(k=ROI, mu=MMR)
            ROI_RideNum.append((ROI, round(PMF*max_rides_num)))
            ROI+=1
        return ROI_RideNum

    def get_driver_accept_prob(self, df):
        """ Model: logistic regression
            df: Drivers' similar Pay-Acceptance data
            Returns the model artifact
        """
        X, y = df['PAY'].values.reshape(-1, 1), df['ACCEPTED'].values.reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
        model = LogisticRegression().fit(X_train, y_train)
        # predictions = model.predict(X_test)
        # mse = mean_squared_error(y_test, predictions)
        # acc = accuracy_score(y_test, predictions)
        # precision = precision_score(y_test, predictions)
        # recall = recall_score(y_test, predictions)
        # print("MSE: ", mse,'\n', "ACC: ", acc,'\n',"precision: ", precision,'\n',"recall: ", recall)
        return model


    def driver_accept_prob(self, model):
        """ model: the trained model from get_driver_accept_prob function
            Returns the pairs of "payment amount" and "drivers acceptance probability" for all levels of payment
        """
        DAP = []
        # y = np.expand_dims(np.array([pay]), 0)
        # DAP.append((pay,model.predict_proba(y)[0][1]))
        y = np.arange(min_pay, max_pay+1, pay_increment)
        # y = np.expand_dims(np.array([pay]), 0)
        probs = model.predict_proba(y.reshape(-1,1))[:,1]
        DAP = list(zip(y, probs))
        return DAP


if __name__ == "__main__":
    # attributes
    months_of_service = 12
    rider_charge = 30
    total_riders = 10000
    rides_monthly_cap = 1000
    lambda_1stMonth = 1
    min_pay = 23
    max_pay = 37
    pay_increment = 7
    # create an object with above attributes
    obj = Pricing_Case_Study(months_of_service, rider_charge, total_riders, rides_monthly_cap,
                             lambda_1stMonth, min_pay, max_pay, pay_increment)

    # get driverAcceptanceData
    df = pd.read_csv('driverAcceptanceData.csv').drop('Unnamed: 0',axis=1)

    # # Descriptive studies
    # obj.descriptive_analysis()

    # Train similar drivers data to get driver's acceptance probability
    model = obj.get_driver_accept_prob(df)


    ride_dist = [] # To collect number of rides based on the amount of payment to driver.
    # Then it will be saved in the Ride_grid dataframe with columns= (Month, ROI, and Pay_X).
    profit_dist = [] # To collect profit based on the amount of payment to driver.
    # Then it will be saved in the Profit_grid dataframe with columns= (Month, ROI, and Pay_X).
    # Get Driver's Acceptance Probability (DAP) based on the pay amount.
    MMR = 1  # Matched Monthly Ride (mu or lambda in Poisson distribution)
    # Note we need to determine MMR for the next month
    DAP = obj.driver_accept_prob(model)  # Returns pairs of "payment amount" and "drivers acceptance probability",
    # using a regression model trained by similar drivers data.

    Ride_grid = []
    Profit_grid = []

    for Month in range(1,3):
        # Get number of rides out of maximum monthly cap for each ROI (Ride Occurance Index).
        if Month == 1:
            # pois_dist_probs returns pairs of "ROI" and "Ride Numbers", using poisson distribution
            # for a specific lambda or mu which is MMR(previous month's matched rides).
            max_rides_num = rides_monthly_cap
            ROI_RideNum = obj.pois_dist_probs(MMR,max_rides_num)
            # Calculate the probability-based profit
            ROI_, RideNum = zip(*ROI_RideNum)
            Pay, DAP_ = zip(*DAP)
            for i in range(1):  # (len(ROI_)): # For now, work with MMR=1 for the subsequent months
                Ride_Numbers = np.round((RideNum)[i] * np.array(list(DAP_)), 0).astype(int)
                profit_Amount = (30 - np.array(list(Pay))) * Ride_Numbers
                ride_dist.append([Month, (i + 1)] + list(Ride_Numbers))
                profit_dist.append([Month, (i + 1)] + list(profit_Amount))
            Ride_grid.append(pd.DataFrame(ride_dist, columns=list(['Month', 'RideOccuranceIndex'] + ['Pay_' + str(i) for i in range(min_pay, max_pay+1, pay_increment)])))
            Profit_grid.append(pd.DataFrame(profit_dist, columns=list(['Month', 'RideOccuranceIndex'] + ['Pay_' + str(i) for i in range(min_pay, max_pay+1, pay_increment)])))
        else:
            for p in range(min_pay, max_pay+pay_increment, pay_increment):
                for i in range(1):  # (len(ROI_)): # For now, work with MMR=1 for the subsequent months
                    cond = (Ride_grid[Month-2]['Month'] == Month - 1) & (Ride_grid[Month-2]['RideOccuranceIndex'] == i+1)
                    max_rides_num = Ride_grid[Month-2].loc[cond, ("Pay_" + str(p))][i]
                    ROI_RideNum = obj.pois_dist_probs(MMR, max_rides_num)
                    ROI_, RideNum = zip(*ROI_RideNum)
                    Pay, DAP_ = zip(*DAP)
                    Ride_Numbers = np.round((RideNum)[i] * np.array(list(DAP_)), 0).astype(int)
                    profit_Amount = (30 - np.array(list(Pay))) * Ride_Numbers
                    ride_dist.append([Month, (i + 1)] + list(Ride_Numbers))
                    profit_dist.append([Month, (i + 1)] + list(profit_Amount))
            Ride_grid.append(pd.DataFrame(ride_dist, columns=list(
                ['Month', 'RideOccuranceIndex'] + ['Pay_' + str(j) for j in
                                                   range(min_pay, max_pay + 1, pay_increment)])))
            Profit_grid.append(pd.DataFrame(profit_dist, columns=list(
                ['Month', 'RideOccuranceIndex'] + ['Pay_' + str(k) for k in
                                                   range(min_pay, max_pay + 1, pay_increment)])))


    print(Profit_grid)

