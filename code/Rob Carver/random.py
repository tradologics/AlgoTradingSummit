
import matplotlib
matplotlib.use("TkAgg")
matplotlib.rcParams.update({'font.size': 22})
from matplotlib.pyplot import hist, plot
import pandas as pd
import numpy as np

# libraries from https://github.com/robcarver17/pysystemtrade
from syscore.dateutils import BUSINESS_DAYS_IN_YEAR, ROOT_BDAYS_INYEAR
from syscore.accounting import accountCurveSingleElementOneFreq as accountCurve

def arbitrary_timeindex(Nperiods, index_start=pd.datetime(2000, 1, 1)):
    """
    For nice plotting, convert a list of prices or returns into an arbitrary pandas time series
    """

    ans = pd.bdate_range(start=index_start, periods=Nperiods)

    return ans


def skew_returns_annualised(annualSR=1.0, want_skew=0.0, voltarget=0.20, size=10000):
    annual_rets = annualSR * voltarget
    daily_rets = annual_rets / BUSINESS_DAYS_IN_YEAR
    daily_vol = voltarget / ROOT_BDAYS_INYEAR

    return skew_returns(want_mean=daily_rets, want_stdev=daily_vol, want_skew=want_skew, size=size)


def skew_returns(want_mean, want_stdev, want_skew, size=10000):
    EPSILON = 0.0000001
    shapeparam = (2 / (EPSILON + abs(want_skew))) ** 2
    scaleparam = want_stdev / (shapeparam) ** .5

    sample = list(np.random.gamma(shapeparam, scaleparam, size=size))

    if want_skew < 0.0:
        signadj = -1.0
    else:
        signadj = 1.0

    natural_mean = shapeparam * scaleparam * signadj
    mean_adjustment = want_mean - natural_mean

    sample = [(x * signadj) + mean_adjustment for x in sample]

    return sample


"""
Do the bootstrap of many random curves
"""

def generate_account_curves(annualSR=1.0, want_skew=0.0, voltarget=0.20,length_backtest_years = 10, number_of_random_curves=1000):
    length_bdays = int(length_backtest_years * BUSINESS_DAYS_IN_YEAR)
    random_curves=[skew_returns_annualised(annualSR=annualSR, want_skew=want_skew, size=length_bdays, voltarget = voltarget)
                   for NotUsed in range(number_of_random_curves)]

    ## Turn into a dataframe

    random_curves_npa=np.array(random_curves).transpose()
    pddf_rand_data=pd.DataFrame(random_curves_npa, index=arbitrary_timeindex(length_bdays), columns=[str(i) for i in range(number_of_random_curves)])

    ## This is a nice representation as well
    acccurves_rand_data=[accountCurve(pddf_rand_data[x], 1.0) for x in pddf_rand_data]

    return acccurves_rand_data

## Get results for various things

## standard deviation
length_backtest_years=10
annualSR=0.5
list_of_vol_targets = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5]
results = []
for voltarget in list_of_vol_targets:
    print(voltarget)
    acccurves_rand_data = generate_account_curves(annualSR=annualSR, voltarget=voltarget,
                                                  length_backtest_years=length_backtest_years)

    drawdown_list = [acc.worst_drawdown() for acc in acccurves_rand_data]

    results.append(    np.median(drawdown_list))

plot(list_of_vol_targets, results)

voltarget = 0.2
length_backtest_years = 10

SR_list = [0,.1,.25,.5,.75,1,1.5,2]
results = []
for annualSR in SR_list:
    print(annualSR)
    acccurves_rand_data = generate_account_curves(annualSR=annualSR, voltarget=voltarget,
                                                  length_backtest_years=length_backtest_years)

    drawdown_list = [acc.worst_drawdown() for acc in acccurves_rand_data]

    results.append(    np.median(drawdown_list))

plot(SR_list, results)

annualSR=0.5
voltarget = 0.2

length_list = [1,2,5,10,20,30]
results = []
for length_backtest_years in length_list:

    print(length_backtest_years)
    acccurves_rand_data = generate_account_curves(annualSR=annualSR, voltarget=voltarget,
                                                  length_backtest_years=length_backtest_years)

    drawdown_list = [acc.worst_drawdown() for acc in acccurves_rand_data]

    results.append(    np.median(drawdown_list))

plot(length_list, results)

length_backtest_years=10
annualSR=0.5
voltarget = 0.2

acccurves_rand_data = generate_account_curves(annualSR=annualSR, voltarget=voltarget, length_backtest_years = length_backtest_years)
drawdown_list = [acc.worst_drawdown() for acc in acccurves_rand_data]

hist(drawdown_list, 100)
np.median(drawdown_list)



voltarget = 0.2
length_backtest_years = 10
SR_list = [0,.1,.25,.5,.75,1,1.5,2]
results = []
for annualSR in SR_list:
    length_bdays = int(length_backtest_years * BUSINESS_DAYS_IN_YEAR)
    print(annualSR)
    acccurves_rand_data = generate_account_curves(annualSR=annualSR, voltarget=voltarget,
                                                  length_backtest_years=length_backtest_years)

    drawdown_list = [acc.worst_drawdown() for acc in acccurves_rand_data]

    results.append(    np.median(drawdown_list))

poss_vol_targets = [voltarget*0.5/-dd for dd in results]
kelly_vol_targets = [sr/2.0 for sr in SR_list]
plot(SR_list, kelly_vol_targets)
plot(SR_list, poss_vol_targets)

length_backtest_years=10
annualSR=0.5
voltarget = 0.20
"""
Do the bootstrap of many random curves
"""

acccurves_rand_data = generate_account_curves(annualSR=annualSR, voltarget=voltarget,
                                              length_backtest_years=length_backtest_years)

drawdown_list = [acc.worst_drawdown() for acc in acccurves_rand_data]

hist(drawdown_list, 100)
max_vol_target = [voltarget*0.5/-dd for dd in drawdown_list]
hist(max_vol_target, bins=50)

sr_vol_target_list = [acc.sharpe()/2.0 for acc in acccurves_rand_data]
hist(sr_vol_target_list)
{"mode":"full","isActive":false}
