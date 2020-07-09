import pandas as pd
import numpy as np
from nsepy import get_history
import datetime
from scipy import stats
from scipy.optimize import minimize
import copy


def get_returns_from_close_as_dataframe(nse_tickers, start_date, end_date):
    """
    Input: list of tickers for which nse data is needed, start date and end date of period.
    Output: two dictionaries with tickers as keys and daily closing prices and daily returns as dataframes
    """
    closing_prices = {}
    returns = {}
    for ticker in nse_tickers:
        print(ticker)
        data = get_history(symbol=ticker, start=start_date, end=end_date)
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
        annualised_vol = returns_df.std()*np.sqrt(annualizing_factor)
        annualized_return = (returns_df+1).prod()**(annualizing_factor/returns_df.shape[0]) -1
        return_2_risk = annualized_return/annualised_vol
        sharpe_ratio = (annualized_return - risk_free_return)/annualised_vol
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
        drawdown_date.append(max_drawdown_df['Max Drawdown'].idxmin())
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
def cppi(risky_assets, start, safe_r=None, risk_free_return=0.0581, floor=0.80, drawdown_constraint=None):
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
    m = 3     #the multiplier
    peak = start

    ## set up some DataFrames for saving intermediate values
    account_history = pd.DataFrame().reindex_like(risky_returns_df)
    risky_w_history = pd.DataFrame().reindex_like(risky_returns_df)
    cushion_history = pd.DataFrame().reindex_like(risky_returns_df)
    
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
    "safe_r": safe_r
    }
    return cppi_result

def summary_stats(returns, riskfree_rate=0.0581):
    """
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    """
    annualised_returns = get_AnnVol_AnnRet_Ret2Risk_Sharpe(returns)
    dd = get_Drawdown(returns)['Max Drawdown in %']
    skew_kurt = get_skewness_kurtosis_isnormallydist(returns)
    the_vars = get_semideviation_var_cvar(returns)
    return pd.concat([annualised_returns, dd, skew_kurt, the_vars], axis=1, sort=False)