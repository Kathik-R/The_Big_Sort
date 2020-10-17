import pandas as pd
import numpy as np
from nsepy import get_history
import datetime
from scipy import stats
from scipy.optimize import minimize
import copy
import math

def compound(r):
    """
    returns the result of compounding the set of returns in r
    """
    return np.expm1(np.log1p(r).sum())

def get_returns_from_close_as_dataframe(nse_tickers, start_date, end_date, index=False):
    """
    Input: list of tickers for which nse data is needed, start date and end date of period.
    Output: two dictionaries with tickers as keys and daily closing prices and daily returns as dataframes
    """
    closing_prices = {}
    returns = {}
    for ticker in nse_tickers:
        print(ticker)
        data = get_history(symbol=ticker, start=start_date, end=end_date,index=index)
        print('done :)')
        closing_prices_df = pd.DataFrame(index=pd.to_datetime(list(data.index), format="%Y-%m-%d"))
        returns_df = pd.DataFrame(index=pd.to_datetime(list(data.index), format="%Y-%m-%d"))
        closing_prices_df[ticker] = data['Close'].to_list()
        returns_df = closing_prices_df.pct_change().drop(list(closing_prices_df.index)[0])
        closing_prices[ticker], returns[ticker] = closing_prices_df, returns_df
    return closing_prices, returns

def get_AnnVol_AnnRet_Ret2Risk_Sharpe(returns, annualizing_factor=260, risk_free_return=0.0581):
    """
    Input: a dictionary of returns for each company, the annualizing factor depending on the frequency of returns, risk free return
    Output: a dataframe with 'Annualized Volatility','Annualized Returns','Return to Risk Ratio','Sharpe Ratio' for all the entities
    """
    stats_df = pd.DataFrame()
    for returns_df in returns.values():
        annualised_vol = returns_df.std()*(annualizing_factor**0.5)
        annualized_return = (returns_df+1).prod()**(annualizing_factor/returns_df.shape[0])-1
        return_2_risk = annualized_return/annualised_vol
        risk_free_return_per_period = (1+risk_free_return)**(1/annualizing_factor)-1
        excess_return = returns_df - risk_free_return_per_period
        ann_excess_return = (excess_return+1).prod()**(annualizing_factor/excess_return.shape[0])-1
        sharpe_ratio = ann_excess_return/annualised_vol
        stats_df = pd.concat([stats_df,pd.concat([annualised_vol, annualized_return, return_2_risk, sharpe_ratio], axis="columns")], axis=0)
    stats_df.columns = ['Annualized Volatility','Annualized Returns','Return to Risk Ratio','Sharpe Ratio']
    return stats_df

def get_Drawdown(returns):
    """
    Takes a time series of asset returns.
    returns a DataFrame with columns for the wealth index, 
    the previous peaks, and the percentage drawdown
    """
    drawdowns = pd.DataFrame(index=list(returns.keys()))
    drawdown_value = []
    drawdown_date = []
    for returns_df in returns.values():
        max_drawdown_df = pd.DataFrame(index=list(returns_df.index))
        max_drawdown_df['Wealth Index'] = 1000*(1+returns_df).cumprod()
        max_drawdown_df['Previous Peaks'] = max_drawdown_df['Wealth Index'].cummax()
        max_drawdown_df['Max Drawdown'] = (max_drawdown_df['Wealth Index'] - max_drawdown_df['Previous Peaks'])/max_drawdown_df['Previous Peaks']
        drawdown_value.append(max_drawdown_df['Max Drawdown'].min()*100)
        drawdown_date.append(max_drawdown_df['Max Drawdown'].astype('float64').idxmin())
    drawdowns['Max Drawdown in %'], drawdowns['Max Drawdown Date'] = [drawdown_value, drawdown_date]
    return drawdowns

def get_skewness_kurtosis_isnormallydist(returns, level=0.01):
    """
    INPUT: 
    The dictionary with all the returns 
    Required level of confidence that the returns are normally distributed (by default 0.01 or 1%)
    
    OUTPUT:
    Dataframe with 'Returns Mean','Returns Median','Skewed?',
    'Skewness','Excess Kurtosis','Normal as per Jarque?' 
    for each company.
    
    NOTES: 
    A distribution is negatively skewed if the median return (most probable) is less than the average return.
    This is because in normal distribution the mean is equal to the median.
    KURTOSIS FOR NORMALLY DISTRIBUTED RETURNS IS 3. Excess Kurtosis is the observed kurtosis minus 3.
    
    Applies the Jarque-Bera test to determine if a Returns are 
    normal or not Test is applied at the 1% level by default
    Returns True if the hypothesis of normality is accepted, False otherwise
    """
    skewness_df = pd.DataFrame()
    jarque_test = []
    for returns_df in returns.values():
        demeaned_ret = returns_df-returns_df.mean()
        stdev = returns_df.std(ddof=0)      ##Delta Degrees of Freedom. The divisor used in calculations is N - ddof, where N represents the number of elements.
        skewness = ((demeaned_ret**3).mean())/(stdev**3)
        kurtosis = ((demeaned_ret**4).mean())/(stdev**4) -3
        jarque_test.append(stats.jarque_bera(returns_df)[1]>level)
        skewness_df = pd.concat([skewness_df,pd.concat([returns_df.mean(), returns_df.median(), returns_df.mean()>returns_df.median(),skewness,kurtosis], axis="columns")],axis=0)
    skewness_df.columns = ['Returns Mean','Returns Median','Negatively Skewed?','Skewness','Excess Kurtosis']
    skewness_df['Normal as per Jarque?'] = jarque_test
    return skewness_df

def get_semideviation_var_cvar(returns, worst_percent=5):
    """
    INPUT:
    the dictionary with returns
    WORST_PERCENT is the percent of most negative returns used to get VaRs
    i.e. returns the number such that that percent of the returns
    fall below the VAR number, and the (100-level) percent are above
    
    OUTPUT:
    dataframe with the semideviation, Historical VaR, Gaussian VaR and CVaR for each return series.
    
    NOTE:
    Semideviation is the std dev of negative returns
    r must be a Series or a DataFrame, else raises a TypeError
    """
    semid_var_cvar_df = pd.DataFrame()
    hist_var = []
    for returns_df in returns.values():
        semidev = returns_df[returns_df<0].std(ddof=0)
        hist_var.append(np.percentile(returns_df, worst_percent)*-1)
        z = stats.norm.ppf(worst_percent/100)
        gaus_var = (returns_df.mean() + z*returns_df.std(ddof=0))*-1

        ## calculate skewness, kurtosis and Cornish-Fisher VAR
        demeaned_ret = returns_df-returns_df.mean()
        stdev = returns_df.std(ddof=0)      ##Delta Degrees of Freedom. The divisor used in calculations is N - ddof, where N represents the number of elements.
        skewness = ((demeaned_ret**3).mean())/(stdev**3)
        kurtosis = ((demeaned_ret**4).mean())/(stdev**4)
        cf_z = (z + (z**2 - 1)*skewness/6 +(z**3 -3*z)*(kurtosis-3)/24 -(2*z**3 - 5*z)*(skewness**2)/36)
        cf_var = -(returns_df.mean() + cf_z*stdev)
        
        #Calculate BeyondVaR or CVaR
        returns_beyond = returns_df <= (np.percentile(returns_df, 5))
        cvar = -(returns_df[returns_beyond].mean())
        
        semid_var_cvar_df = pd.concat([semid_var_cvar_df,pd.concat([semidev, gaus_var, cf_var, cvar], axis="columns")],axis=0)
    semid_var_cvar_df.columns = ['Semi-Deviation', 'Gaussian VaR', 'Cornish-Fisher VaR','Conditional VaR']
    semid_var_cvar_df['Historical VaR'] = hist_var
    return semid_var_cvar_df

def compare_timeseries_lengths(returns):
    '''
    INPUT: dictionary with returns dataframe as values and company as key
    
    OUTPUT: a dataframe with the Start Date, End Date and length of return timeseries for each company
    '''
    start = []
    end = []
    length = []
    for df in returns.values():
        start.append(list(df.index)[0])
        end.append(list(df.index)[-1])
        length.append(len(list(df.index)))
    dates = pd.DataFrame({'Start Date':start,'End Date':end, 'Length': length}, index=list(returns.keys()))
    return dates

def get_covariance_matrix(returns):
    '''
    INPUT: dictionary with returns dataframe as values and company as key
    
    OUTPUT: a single dataframe of the returns of all the companies and a covariance matric of these companies and their returns.
    '''
    combined_df = pd.DataFrame(index=list(returns.values())[0].index)
    for df in returns.values():
         combined_df = pd.concat([combined_df.loc[~combined_df.index.duplicated(keep='first')],df.loc[~df.index.duplicated(keep='first')]], join='inner', axis=1, sort=False)
    return combined_df, combined_df.cov()

def get_portfolio_return(weights, annualized_returns):
    '''
    INPUT: weights and the annualized returns of all companies as a series.
    weights are a numpy array or Nx1 matrix and returns are a numpy array or Nx1 matrix/dataframe
    
    OUTPUT: weighted return of the portfolio
    '''
    return weights.T @ annualized_returns

def get_portfolio_vol(weights, cov_matrix):
    """
    INPUT: weights are a numpy array or N x 1 maxtrix and covmat is an N x N matrix/dataframe
    
    OUTPUT: the volatility of the portfolio.
    
    Computes the vol of a portfolio from a covariance matrix and constituent weights
    """
    return (weights.T @ cov_matrix @ weights)**0.5

def get_efficient_frontier_2assets(returns_2, points_on_frontier=20):
    """
    INPUT: the return dictionary of the 2 assets in the portfolio, 
    the number of points on the frontier for which you wish to calculate the return and risk
    
    OUTPUT: the 2-asset efficient frontier
    """
    weights_2 = [np.array([w, 1-w]) for w in np.linspace(0,1,points_on_frontier)]
    eff_frontier = pd.DataFrame(weights_2, columns=['Wt of '+x for x in list(returns_2.keys())])
    portfolio_return = []
    portfolio_risk = []
    annualized_return_2 = get_AnnVol_AnnRet_Ret2Risk_Sharpe(returns_2)['Annualized Returns']
    combined_returns_df, cov_matrix_2 = get_covariance_matrix(returns_2)
    for weights in weights_2:
        portfolio_return.append(get_portfolio_return(weights, annualized_return_2))
        portfolio_risk.append(get_portfolio_vol(weights, cov_matrix_2))
    eff_frontier['Portfolio Return'], eff_frontier['Portfolio Volatility'] = [portfolio_return, portfolio_risk]
    return eff_frontier

def minimize_vol(returns, target_return):
    """
    INPUT: returns of assets in portfolio
    
    OUTPUT: The frontier weights for a portfolio consisting of
    the input assets
    """
    n_assets = len(list(returns.keys()))
    init_guess = np.repeat(1/n_assets, n_assets)
    bound = ((0.0,1.0),)*n_assets
    
    annualized_return = get_AnnVol_AnnRet_Ret2Risk_Sharpe(returns)['Annualized Returns']
    combined_returns_df, cov_matrix = get_covariance_matrix(returns)
    
    return_is_target = {
        'type': 'eq',
        'args': (annualized_return,),
        'fun': lambda weights, annualized_return: target_return - get_portfolio_return(weights, annualized_return)
    }
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    
    results = minimize(get_portfolio_vol, init_guess,
                      args=(cov_matrix,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,return_is_target),
                       bounds=bound)
    
    return results.x

def get_frontier_weights(returns, num_target_points=50):
    """
    INPUT: returns of assets to be used in portfolio and number of
    points for which you need the weights
    
    OUTPUT: gives a list of returns between the min and max 
    annualised returns among the assets that are to be included
    in the portfolio
    """
    annualized_return = get_AnnVol_AnnRet_Ret2Risk_Sharpe(returns)['Annualized Returns']
    frontier_target_returns = np.linspace(annualized_return.min(), annualized_return.max(), num_target_points)
    frontier_target_weights = [minimize_vol(returns, target_return) for target_return in frontier_target_returns]
    return frontier_target_weights

def plot_efficient_frontier(returns, num_of_front_pts=50, plot_cml=True, plot_ewp=True, plot_gmv=True, risk_free_rate=0.0581):
    """
    Plots the multi-asset efficient frontier
    
    INPUT: returns of assets in portfolio and number of points on
    the frontier that you wish to plot
    
    OUTPUT: risk vs return plot 
    """
    optimal_weights = get_frontier_weights(returns, num_of_front_pts)
    annualized_return = get_AnnVol_AnnRet_Ret2Risk_Sharpe(returns)['Annualized Returns']
    combined_returns_df, cov_matrix = get_covariance_matrix(returns)
    rets = [get_portfolio_return(w, annualized_return) for w in optimal_weights]
    vols = [get_portfolio_vol(w, cov_matrix) for w in optimal_weights]
    
    msr_weights = maximize_sharpe_ratio(returns)
    msr_return = get_portfolio_return(msr_weights, get_AnnVol_AnnRet_Ret2Risk_Sharpe(returns)['Annualized Returns'])
    msr_vol = get_portfolio_vol(msr_weights, cov_matrix)
    
    ef = pd.DataFrame({
        "Returns": rets, 
        "Volatility": vols
    })
    
    axes = ef.plot.line(x="Volatility", y="Returns", style='.-')
    axes.set_xlim(left=0)
    
    if plot_cml:
        cml_x = [0, msr_vol]
        cml_y = [risk_free_rate, msr_return]
        axes.plot(cml_x, cml_y, color="green", marker="o", linestyle="dashed")
        
    if plot_ewp:
        equal_wt_port_weights = np.repeat(1/annualized_return.shape[0], annualized_return.shape[0])
        equal_wt_return = get_portfolio_return(equal_wt_port_weights, annualized_return)
        equal_wt_vol = get_portfolio_vol(equal_wt_port_weights, cov_matrix)
        axes.plot([equal_wt_vol], [equal_wt_return], color="goldenrod", marker="o", markersize=7)
    
    if plot_gmv:
        gmv_weights = global_minimum_vol(returns)
        gmv_return = get_portfolio_return(gmv_weights, annualized_return)
        gmv_vol = get_portfolio_vol(gmv_weights, cov_matrix)
        axes.plot([gmv_vol], [gmv_return], color="midnightblue", marker="o", markersize=7)
    
    return axes

def maximize_sharpe_ratio(returns, risk_free_rate=0.0581):
    """
    INPUT: returns of assets in portfolio and risk free rate of return
    
    OUTPUT: The frontier weights for a portfolio consisting of
    the input assets
    """
    n_assets = len(list(returns.keys()))
    init_guess = np.repeat(1/n_assets, n_assets)
    bound = ((0.0,1.0),)*n_assets
    
    annualized_returns = get_AnnVol_AnnRet_Ret2Risk_Sharpe(returns)['Annualized Returns']
    combined_returns_df, cov_matrix = get_covariance_matrix(returns)
    
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    
    def neg_sharpe_ratio(weights, risk_free_rate, annualized_returns, cov_matrix):
        """
        Return the negative of the Sharpe Ratio
        """
        port_return = get_portfolio_return(weights, annualized_returns)
        port_vol = get_portfolio_vol(weights, cov_matrix)
        neg_sharpe_ratio = -((port_return-risk_free_rate)/port_vol)
        return neg_sharpe_ratio
    
    results = minimize(neg_sharpe_ratio, init_guess,
                       args=(risk_free_rate, annualized_returns, cov_matrix,), 
                       method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1),
                       bounds=bound)
    
    return results.x

def global_minimum_vol(returns, risk_free_rate=0):
    """
    INPUT: returns of assets in portfolio
    
    OUTPUT: Weights of Global Minimum Volatility Portfolio
    """
    n_assets = len(list(returns.keys()))
    init_guess = np.repeat(1/n_assets, n_assets)
    bound = ((0.0,1.0),)*n_assets
    
    annualized_returns = pd.Series([1]*n_assets, index =list(returns.keys())) 
    combined_returns_df, cov_matrix = get_covariance_matrix(returns)
    
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    
    def neg_sharpe_ratio(weights, risk_free_rate, annualized_returns, cov_matrix):
        """
        Return the negative of the Sharpe Ratio
        """
        port_return = get_portfolio_return(weights, annualized_returns)
        port_vol = get_portfolio_vol(weights, cov_matrix)
        neg_sharpe_ratio = -((port_return-risk_free_rate)/port_vol)
        return neg_sharpe_ratio
    
    results = minimize(neg_sharpe_ratio, init_guess,
                       args=(risk_free_rate, annualized_returns, cov_matrix,), 
                       method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1),
                       bounds=bound)
    
    return results.x

#CPPI
def cppi(risky_assets, start, safe_r=None, m = 3, risk_free_return=0.0581, floor=0.80, drawdown_constraint=None):
    """
    Run a backtest of the CPPI strategy, given a set of returns for the risky assets as a dictionary
    Returns a dictionary containing: Asset Value History, wealth if entire principle was invested in risky asset, Risk Budget History, Risky Weight History, CPPI Multiplier, Starting Principal amount, Floor as a percentage of Principal, Risky Asset returns and Safe Asset Returns.
    """
    risky_returns_df, cov_mat = get_covariance_matrix(risky_assets)
    risky_returns_df.reset_index(drop=True, inplace=True)
    
    if safe_r==None:
        safe_r = pd.DataFrame().reindex_like(risky_returns_df)
        safe_r.values[:] = risk_free_return/260 # fast way to set all values to a number
    
    # set up the CPPI parameters
    n_steps = len(list(risky_returns_df.index))
    account_value = start
    floor_value = start*floor
    peak = start

    ## set up some DataFrames for saving intermediate values
    account_history = pd.DataFrame().reindex_like(risky_returns_df)
    risky_w_history = pd.DataFrame().reindex_like(risky_returns_df)
    cushion_history = pd.DataFrame().reindex_like(risky_returns_df)
    floorval_history = pd.DataFrame().reindex_like(risky_returns_df)
    peak_history = pd.DataFrame().reindex_like(risky_returns_df)
    
    for step in range(n_steps):
        if drawdown_constraint is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak*(1-drawdown_constraint)
        cushion = (account_value - floor_value)/account_value
        risky_w = m*cushion
        risky_w = np.minimum(risky_w, 1)
        risky_w = np.maximum(risky_w, 0)
        safe_w = 1-risky_w
        risky_alloc = account_value*risky_w
        safe_alloc = account_value*safe_w

        # recompute the new account value at the end of this step
        account_value = risky_alloc*(1+risky_returns_df.iloc[step]) + safe_alloc*(1+safe_r.iloc[step])

        # save the histories for analysis and plotting
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        account_history.iloc[step] = account_value
        floorval_history.iloc[step] = floor_value
        peak_history.iloc[step] = peak
    
    risky_wealth = start*(1+risky_returns_df).cumprod()
    cppi_result = {
    "Wealth": account_history,
    "Risky Wealth": risky_wealth, 
    "Risk Budget": cushion_history,
    "Risky Allocation": risky_w_history,
    "m": m,
    "start": start,
    "floor": floor,
    "risky_r":risky_returns_df,
    "safe_r": safe_r,
    "drawdown_constraint": drawdown_constraint,
    "peak": peak_history,
    "floor": floorval_history
    }
    return cppi_result

def summary_stats(returns, annualizing_factor=260, risk_free_return=0.0581):
    """
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    """
    annualised_returns = get_AnnVol_AnnRet_Ret2Risk_Sharpe(returns, annualizing_factor=annualizing_factor, risk_free_return=risk_free_return)
    dd = get_Drawdown(returns)['Max Drawdown in %']
    skew_kurt = get_skewness_kurtosis_isnormallydist(returns)
    the_vars = get_semideviation_var_cvar(returns)
    return pd.concat([annualised_returns, dd, skew_kurt, the_vars], axis=1, sort=False)

# Geometric Random Walk
def geo_brown_motion(n_years=10, n_scenarios=10000, mu=0.07, sigma=0.15, steps_per_year=260, initial_price=100):
    """
    Evolution of Geometric Brownian Motion trajectories, such as for Stock Prices
    :param n_years:  The number of years to generate data for
    :param n_paths: The number of scenarios/trajectories
    :param mu: Annualized Drift, e.g. Market Return
    :param sigma: Annualized Volatility
    :param steps_per_year: granularity of the simulation
    :param initial_price: initial value of asset
    :return: a numpy array of n_paths columns and n_years*steps_per_year rows
    Xi is Random Normally Distributed Number.
    """
    dt = 1/steps_per_year
    n_steps = int(n_years*steps_per_year)+1
    returns_plus_1 = np.random.normal(loc=1+mu*dt, scale=sigma*np.sqrt(dt), size=(n_steps, n_scenarios))
    returns_plus_1[0] = 1
    # returns to prices
    prices = initial_price*pd.DataFrame(returns_plus_1).cumprod()
    return prices


def discount(t, ir):
    """
    Compute the price of a pure discount bond that pays a dollar at time period t
    and r is the per-period interest rate
    returns a |t| x |r| Series or DataFrame
    r can be a float, Series or DataFrame
    returns a DataFrame indexed by t
    
    ASSUMPTION: The yield curve is flat that is the IR is the same
    for different horizons
    """
    discounts = pd.Series([(ir+1)**-i for i in t])
    discounts.index = t
    return discounts

def present_value(liabilities, ir):
    """
    Compute the present value of a list of liabilities 
    'liabilities' are indexed by time and the values are 
    amounts of each liability returns the present value of
    the sequence
    """
    dates = liabilities.index
    discounts = discount(dates, ir)
    return discounts.multiply(liabilities, axis='rows').sum()

def funding_ratio(assets, liabilities, ir):
    """
    Computes the funding ratio of a series of liabilities, based on 
    an interest rate and current value of assets
    """
    return present_value(assets, ir)/present_value(liabilities, ir)

def inst_to_ann(r):
    """
    Convert an instantaneous interest rate to an annual interest rate
    """
    return np.expm1(r)

def ann_to_inst(r):
    """
    Convert an instantaneous interest rate to an annual interest rate
    """
    return np.log1p(r)

def cir(n_years = 10, n_scenarios=1, a=0.05, b=0.03, sigma=0.05, steps_per_year=12, r_0=None):
    """
    Generate random interest rate evolution over time using the CIR model
    b and r_0 are assumed to be the annualized rates, not the short rate
    and the returned values are the annualized rates as well
    """
    if r_0 is None: r_0 = b 
    r_0 = ann_to_inst(r_0)
    dt = 1/steps_per_year
    num_steps = int(n_years*steps_per_year) + 1 # because n_years might be a float
    
    shock = np.random.normal(0, scale=np.sqrt(dt), size=(num_steps, n_scenarios))
    rates = np.empty_like(shock)
    rates[0] = r_0

    ## For Price Generation
    h = math.sqrt(a**2 + 2*sigma**2)
    prices = np.empty_like(shock)
    ####

    def price(ttm, r):
        _A = ((2*h*math.exp((h+a)*ttm/2))/(2*h+(h+a)*(math.exp(h*ttm)-1)))**(2*a*b/sigma**2)
        _B = (2*(math.exp(h*ttm)-1))/(2*h + (h+a)*(math.exp(h*ttm)-1))
        _P = _A*np.exp(-_B*r)
        return _P
    prices[0] = price(n_years, r_0)
    ####
    
    for step in range(1, num_steps):
        r_t = rates[step-1]
        d_r_t = a*(b-r_t)*dt + sigma*np.sqrt(r_t)*shock[step]
        rates[step] = abs(r_t + d_r_t)
        # generate prices at time t as well ...
        prices[step] = price(n_years-step*dt, rates[step])

    rates = pd.DataFrame(data=inst_to_ann(rates), index=range(num_steps))
    ### for prices
    prices = pd.DataFrame(data=prices, index=range(num_steps))
    ###
    return rates, prices

def bond_cash_flows(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12):
    """
    Returns the series of cash flows generated by a bond,
    indexed by the payment/coupon number
    """
    n_coupons = round(maturity*coupons_per_year)
    coupon_amt = principal*coupon_rate/coupons_per_year
    coupon_times = np.arange(1, n_coupons+1)
    cash_flows = pd.Series(data=coupon_amt, index=coupon_times)
    cash_flows.iloc[-1] += principal # add the principal to the last payment
    return cash_flows
    
def bond_price(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12, discount_rate=0.03):
    """
    Computes the price of a bond that pays regular coupons until maturity
    at which time the principal and the final coupon is returned
    This is not designed to be efficient, rather,
    it is to illustrate the underlying principle behind bond pricing!
    If discount_rate is a DataFrame, then this is assumed to be the rate on each coupon date
    and the bond value is computed over time.
    i.e. The index of the discount_rate DataFrame is assumed to be the coupon number
    """
    if isinstance(discount_rate, pd.DataFrame):
        pricing_dates = discount_rate.index
        prices = pd.DataFrame(index=pricing_dates, columns=discount_rate.columns)
        for t in pricing_dates:
            prices.loc[t] = bond_price(maturity-t/coupons_per_year, principal, coupon_rate, coupons_per_year,discount_rate.loc[t])
        return prices
    else: # base case ... single time period
        if maturity <= 0: return principal+principal*coupon_rate/coupons_per_year
        cash_flows = bond_cash_flows(maturity, principal, coupon_rate, coupons_per_year)
        return present_value(cash_flows, discount_rate/coupons_per_year)

def macaulay_duration(flows, discount_rate):
    """
    Computes the Macaulay Duration of a sequence of cash flows, given a per-period discount rate
    """
    discounted_flows = discount(flows.index, discount_rate)*flows
    weights = discounted_flows/discounted_flows.sum()
    return np.average(flows.index, weights=weights)

def match_durations(cf_t, cf_s, cf_l, discount_rate):
    """
    Returns the weight W in cf_s that, along with (1-W) in cf_l will have an effective
    duration that matches cf_t
    """
    d_t = macaulay_duration(cf_t, discount_rate)
    d_s = macaulay_duration(cf_s, discount_rate)
    d_l = macaulay_duration(cf_l, discount_rate)
    return (d_l - d_t)/(d_l - d_s)

def bond_total_return(monthly_prices, principal, coupon_rate, coupons_per_year):
    """
    Computes the total return of a Bond based on monthly bond prices and coupon payments
    Assumes that dividends (coupons) are paid out at the end of the period (e.g. end of 3 months for quarterly div)
    and that dividends are reinvested in the bond
    """
    coupons = pd.DataFrame(data = 0, index=monthly_prices.index, columns=monthly_prices.columns)
    t_max = monthly_prices.index.max()
    pay_date = np.linspace(12/coupons_per_year, t_max, int(coupons_per_year*t_max/12), dtype=int)
    coupons.iloc[pay_date] = principal*coupon_rate/coupons_per_year
    total_returns = (monthly_prices + coupons)/monthly_prices.shift()-1
    return total_returns.dropna()

def bt_mix(r1, r2, allocator, **kwargs):
    """
    Runs a back test (simulation) of allocating between a two sets of returns
    r1 and r2 are T x N DataFrames or returns where T is the time step index and N is the number of scenarios.
    allocator is a function that takes two sets of returns and allocator specific parameters, and produces
    an allocation to the first portfolio (the rest of the money is invested in the GHP) as a T x 1 DataFrame
    Returns a T x N DataFrame of the resulting N portfolio scenarios
    """
    if not r1.shape == r2.shape:
        raise ValueError("r1 and r2 should have the same shape")
    weights = allocator(r1, r2, **kwargs)
    if not weights.shape == r1.shape:
        raise ValueError("Allocator returned weights with a different shape than the returns")
    r_mix = weights*r1 + (1-weights)*r2
    return r_mix

def fixedmix_allocator(r1, r2, w1, **kwargs):
    """
    Produces a time series over T steps of allocations between the PSP and GHP across N scenarios
    PSP and GHP are T x N DataFrames that represent the returns of the PSP and GHP such that:
     each column is a scenario
     each row is the price for a timestep
    Returns an T x N DataFrame of PSP Weights
    """
    return pd.DataFrame(data = w1, index=r1.index, columns=r1.columns)

def terminal_values(rets):
    """
    Computes the terminal values from a set of returns supplied as a T x N DataFrame
    Return a Series of length N indexed by the columns of rets
    """
    return (rets+1).prod()

def terminal_stats(rets, floor = 0.8, cap=np.inf, name="Stats"):
    """
    Produce Summary Statistics on the terminal values per invested dollar
    across a range of N scenarios
    rets is a T x N DataFrame of returns, where T is the time-step (we assume rets is sorted by time)
    Returns a 1 column DataFrame of Summary Stats indexed by the stat name 
    "mean": average terminal wealth,
    "std" : deviation of terminal wealths from mean,
    "breach_count": count of floor breaches,
    "reach_count": count of cap breach,
    "p_breach": probability of breach,
    "e_short":average value of deviation from floor if floor is being breached,
    "p_reach": probability of reach,
    "e_surplus": average value of deviation from cap if cap is being breached
    """
    terminal_wealth = terminal_values(rets)
    breach = terminal_wealth < floor
    reach = terminal_wealth >= cap
    breach_count = breach.sum()
    reach_count = breach.sum()
    p_breach = breach.mean() if breach.sum() > 0 else 'N/A'
    p_reach = reach.mean() if reach.sum() > 0 else 'N/A'
    e_short = (floor-terminal_wealth[breach]).mean() if breach.sum() > 0 else 'N/A'
    e_surplus = (cap-terminal_wealth[reach]).mean() if reach.sum() > 0 else 'N/A'
    sum_stats = pd.DataFrame.from_dict({
        "mean": terminal_wealth.mean(),
        "std" : terminal_wealth.std(),
        "breach_count": breach_count,
        "reach_count": reach_count,
        "p_breach": p_breach,
        "e_short":e_short,
        "p_reach": p_reach,
        "e_surplus": e_surplus
    }, orient="index", columns=[name])
    return sum_stats

def glidepath_allocator(r1, r2, start_glide=1, end_glide=0.0):
    """
    Allocates weights to r1 starting at start_glide and ends at end_glide
    by gradually moving from start_glide to end_glide over time
    """
    n_points = r1.shape[0]
    n_col = r1.shape[1]
    path = pd.Series(data=np.linspace(start_glide, end_glide, num=n_points))
    paths = pd.concat([path]*n_col, axis=1)
    paths.index = r1.index
    paths.columns = r1.columns
    return paths

def floor_allocator(psp_r, ghp_r, floor, zc_prices, m=3):
    """
    Allocate between PSP and GHP with the goal to provide exposure to the upside
    of the PSP without violating the floor.
    Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple
    of the cushion in the PSP
    Returns a DataFrame with the same shape as the psp/ghp representing the weights in the PSP
    """
    if zc_prices.shape != psp_r.shape:
        raise ValueError("PSP and ZC Prices must have the same shape")
    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1, n_scenarios)     #initialize account value
    floor_value = np.repeat(1, n_scenarios)       #initialize floor value
    w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)     #create empty weights log over time
    for step in range(n_steps):
        floor_value = floor*zc_prices.iloc[step] ## PV of Floor assuming today's rates and flat YC
        cushion = (account_value - floor_value)/account_value
        psp_w = (m*cushion).clip(0, 1) # clip keeps the weight between 0 and 1
        ghp_w = 1-psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        # recompute the new account value at the end of this step
        account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        w_history.iloc[step] = psp_w
    return w_history

def drawdown_allocator(psp_r, ghp_r, maxdd, m=3):
    """
    Allocate between PSP and GHP with the goal to provide exposure to the upside
    of the PSP without going violating the floor.
    Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple
    of the cushion in the PSP
    Returns a DataFrame with the same shape as the psp/ghp representing the weights in the PSP
    """
    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1, n_scenarios)     #initialize account value
    floor_value = np.repeat(1, n_scenarios)       #initialize floor value
    ### For MaxDD
    peak_value = np.repeat(1, n_scenarios)
    w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    for step in range(n_steps):
        ### For MaxDD
        floor_value = (1-maxdd)*peak_value ### Floor is based on Prev Peak
        cushion = (account_value - floor_value)/account_value
        psp_w = (m*cushion).clip(0, 1) # same as applying min and max
        ghp_w = 1-psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        # recompute the new account value at the end of this step
        account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        ### For MaxDD
        peak_value = np.maximum(peak_value, account_value) ### For MaxDD
        w_history.iloc[step] = psp_w
    return w_history

import statsmodels.api as sm
def regress(dependent_variable, explanatory_variables, alpha=True):
    """
    Runs a linear regression to decompose the dependent variable into the explanatory variables
    alpha as TRUE implies that there is an alpha return from the Asset Manager as an explainatory variable
    returns an object of type statsmodel's RegressionResults on which you can call
       .summary() to print a full summary
       .params for the coefficients
       .tvalues and .pvalues for the significance levels
       .rsquared_adj and .rsquared for quality of fit
    """
    if alpha:
        explanatory_variables = explanatory_variables.copy()
        explanatory_variables["Alpha"] = 1
    
    lm = sm.OLS(dependent_variable, explanatory_variables).fit()
    return lm

def tracking_error(r_a, r_b):
    """
    Returns the Tracking Error between the two return series
    """
    return np.sqrt(((r_a - r_b)**2).sum())

def portfolio_tracking_error(weights, ref_r, bb_r):
    """
    returns the tracking error between the reference returns
    and a portfolio of building block returns held with given weights
    """
    return tracking_error(ref_r, (weights*bb_r).sum(axis=1))

def style_analysis(dependent_variable, explanatory_variables):
    """
    Returns the optimal weights that minimizes the Tracking error between
    a portfolio of the explanatory variables and the dependent variable
    """
    n = explanatory_variables.shape[1]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    solution = minimize(portfolio_tracking_error, init_guess,
                       args=(dependent_variable, explanatory_variables,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    weights = pd.Series(solution.x, index=explanatory_variables.columns)
    return weights

def weight_ew(r, cap_weights=None, max_cw_mult=None, microcap_threshold=None, **kwargs):
    """
    Returns the weights of the EW portfolio based on the asset returns "r" as a DataFrame
    If supplied a set of capweights and a capweight tether, it is applied and reweighted 
    """
    n = len(r.columns)
    ew = pd.Series(1/n, index=r.columns)
    if cap_weights is not None:
        cw = cap_weights.loc[r.index[0]] # starting cap weight
        ## exclude microcaps
        if microcap_threshold is not None and microcap_threshold > 0:
            microcap = cw < microcap_threshold
            ew[microcap] = 0
            ew = ew/ew.sum()
        #limit weight to a multiple of capweight
        if max_cw_mult is not None and max_cw_mult > 0:
            ew = np.minimum(ew, cw*max_cw_mult)
            ew = ew/ew.sum() #reweight
    return ew

def weight_cw(r, cap_weights, **kwargs):
    """
    Returns the weights of the CW portfolio based on the time series of capweights
    """
    w = cap_weights.loc[r.index[0]]
    return w/w.sum()

def backtest_ws(r, estimation_window=60, weighting=weight_ew, **kwargs):
    """
    Backtests a given weighting scheme, given some parameters:
    r : asset returns to use to build the portfolio
    estimation_window: the window to use to estimate parameters. Eg: backtest based on previous 60 day average returns data
    weighting: the weighting scheme to use, must be a function that takes "r", and a variable number of keyword-value arguments
    """
    n_periods = r.shape[0]
    # return windows
    windows = [(start, start+estimation_window) for start in range(n_periods-estimation_window+1)]
    weights = [weighting(r.iloc[win[0]:win[1]], **kwargs) for win in windows]
    # convert to DataFrame
    weights = pd.DataFrame(weights, index=r.iloc[estimation_window-1:].index, columns=r.columns)
    returns = (weights * r).sum(axis="columns",  min_count=1) #mincount is to generate NAs if all inputs are NAs
    return returns
