# Useful Transformations

def calc_log_ratio(s1, s2, eps=0):
    return np.log(s1 + eps) - np.log(s2 + eps)

def relative_series_log(series, lookback=20, eps=0):
    s1 = series
    s2 = series.rolling(lookback, min_periods=2).mean().shift(1)
    return calc_log_ratio(s1, s2, eps)

# Price Direction

def calc_log_return(ohlc):
    x1 = ohlc['open'].shift(-1)
    x2 = ohlc['open'].shift(-2)
    log_return = calc_log_ratio(x1, x2)
    
    return log_return

def calc_mpe(ohlc, lookforward=0):
    max_high = (
        ohlc["high"]
        .rolling(lookforward + 1)
        .max()
        .shift(-lookforward)
    )
    mpe = max_high - ohlc["open"]

    return mpe

def calc_mne(ohlc, lookforward=0):
    min_low = (
        ohlc["low"]
        .rolling(lookforward + 1)
        .min()
        .shift(-lookforward)
    )
    mne = ohlc["open"] - min_low

    return mne

def calc_edge_ratio_log(ohlc, lookforward=1, eps=1e-6, max_val=25):
    mpe = calc_mpe(ohlc, lookforward)
    mne = calc_mne(ohlc, lookforward)
    edge_ratio_log = calc_log_ratio(mpe, mne, eps=eps).clip(-max_val, max_val)

    return edge_ratio_log

# Range / Volatility

def calc_true_range(ohlc):
    h, l, c = split_ohlc(ohlc)[1:4]
    method_1 = h - l
    method_2 = (h - c.shift(1)).abs()
    method_3 = (l - c.shift(1)).abs()
    true_range = pd.concat((method_1, method_2, method_3), axis=1).max(axis=1)

    return true_range

def calc_average_true_range(ohlc, lookback):
    true_range = calc_true_range(ohlc)
    average_true_range = true_range.rolling(lookback).mean()

    return average_true_range

def ATR_perc(ohlc, lookback):
    true_range = TR(ohlc)
    true_range_percent = true_range / ohlc["open"]
    average_true_range_percent = true_range_percent.rolling(lookback).mean()
    return average_true_range_percent

def calc_tape_length(price_series):
    abs_diffs = price_series.diff().abs()
    tape_length = abs_diffs.sum()

    return tape_length

def calc_total_move(price_series):
    total_move = abs(price_series[-1] - price_series[0])

    return total_move

# Trend

def calc_efficiency_ratio(series):
    total_diff = abs(series[-1] - series[0])
    sum_diffs = series.diff().abs().sum()
    return total_diff / sum_diffs

def calc_moving_average_dominance(series, lookback):
    ema = series.ewm(span=lookback).mean()
    over_ema = series > ema
    mad_raw = over_ema.sum() / len(series)
    moving_average_dominance = max(mad_raw, 1-mad_raw)
    
    return moving_average_dominance
