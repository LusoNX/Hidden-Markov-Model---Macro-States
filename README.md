# Hidden-Markov-Model---Macro-States
Practical application of a Hidden Markov Model used for the prediction of macro scenarios. 


# Introduction 

The following project demonstrates a practical application of an Hidden Markov Model (HMM) used in the estimation of macro states. Markets work in regimes and regimes conditioned asset classes performance. As such, its extremely important for a portfolio manager to comprehend in which regime he is trading and allocate the assets in the 
However the precise categorization of these regimes is not straightfoward. 
Firstly, there is the question of "What do the regimes mean?". 
_Bullish_ regimes mean periods in which market performance is expected to trend positively. _Bearish_ regimes is when they are expected to trend negatively. And _neutral_ regimes is when they are expected to trade lateraly (ranging markets). Market can be also associated to their level of volatility. Option traders may not be interested in the direction of the market, but rather on its volatility.

Secondly, "How many regimes?"

Thirdly, "What separates them?"

This project is inspired in [CITE LITERATURE] papers. 

Through a simple combination of simply 4 features, this project explores the use of a Hidden Markov Model in defining a set of hidden states (market regimes). 
Each hidden state is determined using as inputs the the VIX, market returns (SP500), CPI, and industrial production index (INDPRO). These variables help identify and classify the underlying market conditions that the hidden state captures. In other words, each hidden state encapsulates a historical phase where these variables exhibited similar behaviors, allowing for the classification of market conditions into regimes with shared structural characteristics.

The hidden states are used to construct return vectors and covariance matrices, embedding conditional information specific to the selected asset classes for each particular market regime.
Portfolios are optimized based on these inputs and backtested, in order to evaluate the robustness of the model, and whether added value can be made through this rebalancing approach. 


# The Model

The project uses the library hmmlearn in order to implement the HMM. 

The model uses as input 4 variables. 
1. Yearly Inflation
2. Monthly Industrial Production Index (INDPRO) change (%)
3. SP500 monthly market returns
4. VIX close value.

Frequency of the model is monthly and values are scaled through an expanding window before the fitting of the model. 
We shift the model by 2 in order for later backtesting and to prevent look ahead bias. Since the macro data used for the given month is only reported in the subsequent month and typically only after the 15 of the next month, a shift of 2 is a conservative approach
in order to account for the lag of the data for a given month. The market returns and VIX are only shifted by 1, to represent 

''' python 

    def MacroVariables(self, _frequency):
        df_INDPRO = pd.read_csv(rf"{os.getcwd()}\DATA\INDPRO.csv", index_col=0)
        df_INDPRO = df_INDPRO.rename(columns={"Value": "Real GDP"})
        df_INDPRO.index = pd.to_datetime(df_INDPRO.index)
    
        df_CPI = pd.read_csv(rf"{os.getcwd()}\DATA\CPI.csv", index_col=0)
        df_CPI = df_CPI.rename(columns={"Value": "CPI"})
        df_CPI.index = pd.to_datetime(df_CPI.index)
    
        df_VIX = pd.read_csv(rf"{os.getcwd()}\DATA\VIX.csv", index_col=0)
        df_VIX = df_VIX.rename(columns={"Value": "Market Volatility"})
        df_VIX.index = pd.to_datetime(df_VIX.index)
    
        # Uses yfinance to get S&P 500 data
        df_sp500 = yf.Ticker("^GSPC")
        df_sp500 = df_sp500.history(period="max")
        df_sp500.index = df_sp500.index.date
        df_sp500.index = pd.to_datetime(df_sp500.index)
        df_sp500 = df_sp500[["Close"]]
        df_sp500 = df_sp500.resample(_frequency).last()
    
        # Set Frequencies
        if _frequency == "M":
            df_INDPRO = df_INDPRO.pct_change(1).iloc[1::]
            df_INDPRO.index = pd.to_datetime(df_INDPRO.index) + pd.offsets.MonthEnd(0)
    
            df_CPI = df_CPI.pct_change(12).iloc[12::]
            df_CPI.index = pd.to_datetime(df_CPI.index) + pd.offsets.MonthEnd(0)
    
            df_sp500["Market Returns"] = df_sp500["Close"].pct_change(1).iloc[1::]
            df_sp500 = df_sp500.resample(_frequency).last()
    
            df_VIX.index = pd.to_datetime(df_VIX.index)
            df_VIX = df_VIX.resample(_frequency).last()
    
            df = df_INDPRO.merge(df_CPI, how="inner", right_index=True, left_index=True)
        df = df.merge(df_sp500, how="inner", right_index=True, left_index=True)
        df = df.merge(df_VIX, how="inner", right_index=True, left_index=True)
    
        df = df.sort_index()
        df = df[["Real GDP", "CPI", "Market Returns", "Market Volatility"]]
        df[["Real GDP", "CPI"]] = df[["Real GDP", "CPI"]].shift(2)
    
        # For Market Volatility and Returns we can simply shift by 1.
        df[["Market Returns", "Market Volatility"]] = df[["Market Returns", "Market Volatility"]].shift(1)
        df = df.iloc[2::]


The helper_optimal_states method helps determine the optimal number of hidden states for the HMM by fitting models with varying state counts (between 2 and 10), calculating the Bayesian Information Criterion (BIC) for each model, and selecting the model with the lowest BIC score. This approach ensures the best-fitting model based on both likelihood and complexity, guiding the selection of the optimal number of hidden states.

''' python 

    def helper_optimal_states(self,observed_data):

        # Helper function used to determined the optimal number of states for the scenario prediction.
        bic_scores = []
        models = []
        min_states = 2
        max_states =10
        # Loop over different numbers of states
        for n_states in range(min_states, max_states + 1):
            # Create and fit the HMM model
            model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag")#,random_state = 42)
            model.fit(observed_data)
            # Calculate the log likelihood
            log_likelihood = model.score(observed_data)
            # Number of parameters
            n_params = n_states * (n_states - 1) + 2 * observed_data.shape[1] * n_states

            # BIC calculation
            bic = -2 * log_likelihood + n_params * np.log(len(observed_data))

            # Store the BIC score and model
            bic_scores.append(bic)
            models.append(model)

        # Find the optimal number of states with the lowest BIC score
        optimal_n_states = np.argmin(bic_scores) + min_states
        print(f"The optimal number of states are: {optimal_n_states}")
        return optimal_n_states



Hidden states are estimated on a expanding window method, where at each month, new data is fitted into the previously trained model. A _test_date_ is set to define the  batch of data used for the initial training iteration. 

 ''' python 

     def HMMScenariosTrainTest(self,hmm_model,_frequency,test_date):

        scaler = StandardScaler()
        if hmm_model == "Macro":
            df = self.MacroVariables(_frequency)
                            

        test_date_adj = pd.to_datetime(test_date)
        train_df = df[df.index < test_date_adj]

        #optimal_nr_scenarios = self.helper_optimal_states(scaler.fit_transform(train_df)) # Helper function to determine the optimal nr of scenarios 
        optimal_scenarios = {"Macro":4,"Market":4}

        test_df =df[df.index >= test_date_adj]
        train_df  = train_df.sort_index()
        test_df = test_df.sort_index()
        dict_of_inputs = {}
            
        
        # Initialized an empty model. Required to maintain consistency over each new iteration 
        prev_model = None
        nr_of_scenario_failures = 0 


The model re-fiting is accomplished by setting the model parameters of the previous iteration in the subsequent iteration, re-fitting the model and repeating the process. Over time, the transition matrix and state means converge towars their long term equilibrium, but the process ensures that "future" data is not embedded in the fitting of the model. 

''' python 

    for i in range(len(test_df)+1):
        df_new = pd.concat([train_df,test_df.iloc[0:0+i]])
        _date = df_new.index.values[-1]
        _date =_date + pd.offsets.MonthEnd(0)
        date_str = _date.strftime("%Y-%m-%d")
        
    
        if i == 0:
            scaled_values = scaler.fit_transform(df_new)
            model = hmm.GaussianHMM(n_components=optimal_scenarios[hmm_model], covariance_type="full",init_params="mc",params="stmcm",n_iter=10000,random_state=42)
            n_components = model.n_components
            model.startprob_ = np.full(n_components, 1 / n_components)
            model.transmat_ = np.full((n_components, n_components), 1 / n_components)
    
        else:
            scaled_values = scaler.fit_transform(df_new)
            model = hmm.GaussianHMM(n_components=optimal_scenarios[hmm_model], covariance_type="full",params="stmcm",init_params="",n_iter=10000,random_state = 42)           
            model.startprob_  = prev_startprob
            model.transmat_ = prev_transmat
            model.means_ = prev_means



# Hidden States Statistics

By default, hidden states are utilized to partition the higher-dimensional space into distinct regimes, but (for now) they dont have any representation. 
To classify the hidden states we can evaluate the statistics of the features used in the HMM model determination.
We can achieve this by running "_PlotResultsGIT.py_". 
The figure below shows a 3-4D space (volatility represented by the size of the dots) of the 4 input variables and their respective position


### 4-D plot of the four features by HS
![image](https://github.com/user-attachments/assets/1305c800-d2a0-4c9d-b42e-4311c3d0740c)


We can observe that Hidden State (HS) 1 is characteristic of periods of higher inflation while the remaining dots are more of low to normal volatility. 
Furthermore HS 2 is characterized of periods of lower volatility and an average market return, suggesting moments of trending bull markets. 
HS 4 is characterized by periods of lower returns and also lower INDPRO, associated with periods of economic contraction 
Finally, HS 3 is characterized by the periods with the highest level of the VIX. 


### Mean values of the four features by HS
![image](https://github.com/user-attachments/assets/dc948cb1-896e-46bb-96f2-51fe83be6c85)

Furthermore we can inspect how these HS's relate with market performance over time, by plotting theem in relation to the SP500 performance. 


### HS over the SP500
![hidden states](https://github.com/user-attachments/assets/c4b52539-721d-40a9-a552-949981a0eaea)

As expected, HS 2 periods are typical of trending markets. HS 3 associated with periods of higher volatility, with a positive or negative direction. HS 1, related with a higher level of inflation is characterized of periods of lower market performance.


# Portfolio Optimization

The optimization process consists in finding the optimal weights for a portfolio combination (overtime) for 4 factors, namely _Momentum_, _Low Volatility_, _Value_ and _Growth_.

## Mean Returns 
Means and covariance matrices are computed based on returns contained within each hidden state. The goal is in **finding the optimal portfolios conditioned to their HS**. 
For the mean, i compute the Exponential Weighted Moving Average (EWMA) of returns using a 6 week span (for weekly testing) and 3 month span (for monthly testing). EWMA is used to smooth the returns  by giving more weight to recent observations, making it sensitive to recent changes.



'''python 

	_span_dict = {"W":6,"M":3}
	for _scenario in df_merge["hidden_states"].unique():
                df_merge_scenario = df_merge[df_merge["hidden_states"] == _scenario]
                df_merge_D_scenario =df_merge_D[df_merge_D["hidden_states"] == _scenario]
                df_merge_scenario = df_merge_scenario.loc[~(df_merge_scenario == 0).any(axis=1)]
                df_merge_scenario.iloc[:,:-1] = df_merge_scenario.iloc[:,:-1].ewm(span=_span_dict[price_frequency], adjust=False).mean()

## Covariance of Returns

Covariance Matrix of the returns is calculated using Random Matrix Theory (RMT), through the Marchenko-Pastur theorem and consists in finding the most relevant eigenvalues of the correlation matrix and filtering out for the most significant eigenvalues, reducing the amount of noise in the data. Furthermore, the approach eliminates multicolinearity, since the retrieved eigenvectors are uncorrelated with one another, further improving the final matrix. 

'''python 

	def helper_covariance_parameters(self,DF,cov_type,_freq):
	        from pypfopt import risk_models
	        df = DF.copy()
	        dict_freq = {"D":250,"W":52,"M":12}
	
	        if cov_type == "normal":
	            port_cov = risk_models.sample_cov(df,returns_data = True, frequency = dict_freq[_freq])
	        elif cov_type == "shrinkage":
	            port_cov = risk_models.CovarianceShrinkage(df,returns_data = True, frequency = dict_freq[_freq]).ledoit_wolf()
	        
	        elif cov_type == "random_matrix":
	            #Model that employs the Random Matrix format where eigen values are removed using the rule of  Marchenko and Pasteur
	            
	            correlation_matrix = df.corr()
	            eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
	            q = len(df) / len(df.columns)  # λ = m/n
	            sigma = 1 
	            lambda_plus = sigma**2 * (1 + (1/np.sqrt(q)))
	            lambda_minus = sigma**2 * (1 - (1/np.sqrt(q)))
	
	            # Step 4: Filter Eigenvalues (keep only significant ones)
	            filtered_eigenvalues = np.where(eigenvalues > lambda_plus, eigenvalues, 0)
	            #print(eigenvalues)
	            #print(eigenvectors)
	            
	            filtered_correlation_matrix = (eigenvectors @ np.diag(filtered_eigenvalues) @ eigenvectors.T)
	            filtered_correlation_matrix[np.diag_indices_from(filtered_correlation_matrix)] = 1
	            volatilities = np.std(df, axis=0)
	
	            # Rescale the filtered correlation matrix by the volatilities
	            filtered_covariance_matrix = filtered_correlation_matrix * np.outer(volatilities, volatilities)  
	            port_cov = pd.DataFrame(filtered_covariance_matrix, index=list(df.columns), columns=list(df.columns))
	                                
	    
	        return port_cov


## Optimization Function 

For the optimization process, a custom Module is created _CustomOptimizer.py_ in order to solve for the optimal portfolio, that take the mean returns and covariance matrix as inputs. 

''' python 

	def solve_sharpe_ratio(self, RET, COV, risk_free_rate, risk_target_value, lambda_reg, max_weight,min_weight):
	        # Number of assets
	        n = len(RET)
	        
	        # Define the variable: weights of the assets in the portfolio
	        w = cp.Variable(n)
	        
	        # Define the expected return of the portfolio
	        portfolio_return = RET @ w
	        
	        # Define the risk (variance) of the portfolio
	        portfolio_variance = cp.quad_form(w, COV)
	        l2_regularization = lambda_reg * cp.sum_squares(w)
	        
	        # Define the objective: maximize the expected return
	        objective = cp.Maximize(portfolio_return - l2_regularization)
	        
	        # Define the constraints
	        constraints = [
	            cp.sum(w) == 1,  # Sum of weights equals 1
	            w >= 0,          # Long-only constraint
	            portfolio_variance <= risk_target_value  # Variance constraint
	        ]
	        
	        # Add maximum weight constraint if provided
	        if max_weight is not None:
	            constraints += [w <= max_weight]
	        
	        # Define the problem
	        problem = cp.Problem(objective, constraints)
	        problem.solve(solver=cp.ECOS, qcp=True)  # Indicate that the problem is a QCP
	        
	        return w.value, portfolio_return.value, portfolio_variance.value

CVXPY is used in the construction of the optimizer. The optimization solves for the maximization of the portfolio_return adjusted by a ridge penalty term - the L2 regularization norm. This prevents the model of finding corner solution, ensuring a higher degree of diversification across the optimizations. 

However, the main objective is not in maximizing returns, but rather the sharpe ratio of the portfolio. But since CVXPY is not suitable to handle non-convext expression, a workaround is done, by setting a given level of volatility, defined by the parameter _risk_target_value_, and iterating over various level of volatilities in the main function, that calls out the function. 

''' python 

	def optimize_sharpe_ratio(self, RET, COV, risk_free_rate, lambda_reg, max_weight=None,min_weight = None,long_short=False):
	        risk_targets = np.linspace(0.01, 0.30, 30)
	        risk_targets = risk_targets**2  # Square the risk targets to represent variance targets
	
	        # Placeholder for the best Sharpe ratio and corresponding weights
	        best_sharpe_ratio = -np.inf
	        best_weights = None
	        
	        # Iterate over the given risk targets
	        for risk_target_value in risk_targets:
	            if long_short == False:
	                weights, expected_return, variance = self.solve_sharpe_ratio(
	                    RET, COV, risk_free_rate, risk_target_value, lambda_reg, max_weight,min_weight
	                )
	            else:
	                weights, expected_return, variance = self.solve_sharpe_ratio_LS(
	                    RET, COV, risk_free_rate, risk_target_value, lambda_reg, max_weight,min_weight
	                )
	                        
	            if weights is not None:
	                # Calculate the Sharpe ratio
	                risk = np.sqrt(variance)  # Use standard deviation (sqrt of variance)
	                sharpe_ratio = (expected_return - risk_free_rate) / risk
	                
	                # Update the best Sharpe ratio and corresponding weights if current Sharpe ratio is better
	                if sharpe_ratio > best_sharpe_ratio:
	                    best_sharpe_ratio = sharpe_ratio
	                    best_weights = weights
	        
	        return best_weights


The portfolio that maximizes the sharpe ratio is selected. The following figure ilustrates the weights over times and their respective concentration in the portfolio - measured using the Herfindahl-Hirschman Index (HHI) approach

### Optimal Weights Overtime
![image](https://github.com/user-attachments/assets/c1956250-6d89-4c32-894f-56da5a717d5f)

### HHI - Portfolio Concentration 

![image](https://github.com/user-attachments/assets/ff575e56-fc34-4991-a062-2837a73e6e81)


As we can observe, the portefolio is relatively concentrated, but spread out over the factors over extended periods of time, with some exceptions where HHI surpasses the 0.7 band. 


# Backtest

For the backtest, weights are shifted by 1, to account for the previous week/month return in the calculation of the current week/month performance . However, this approach implicitly assumes that positions are opened at the closing price of the previous week/month.
Thus, for each given point in time, we estimate the HS, using the four (past) values of the 4 input variables  (CPI, INDPRO, VIX and Market Returns). From this, we estimate the optimal portfolios, conditioned to that HS and shift them by 1. This prevents the "look ahead bias" at both scenario estimation and performance estimation. 

''' python
	def backtest_performance(self,DF_W,_frequency):
	        import matplotlib.pyplot as plt
	        df_w = DF_W.copy()
	        df_w = df_w.sort_index()
	        df_prices = pd.read_csv(rf"{self.cur_dir}\DATA\PRICE_DATA.csv",index_col = 0)
	        df_prices.index = pd.to_datetime(df_prices.index)
	        df_prices = df_prices.resample(_frequency).last()
	        df_prices =df_prices.pct_change()
	        df_prices = df_prices.iloc[1::]
	
	        # Shift 1 month in order to match the weights estimate at a given data with the subsequent performance of the respective factor
	        df_w = df_w.shift(1)
	        df_w = df_w.iloc[1::]
	        df_w.index = pd.to_datetime(df_w.index)
	        df_w = df_w.resample(_frequency).last()
	        df_prices = df_prices[(df_prices.index >= df_w.index.values[0]) & (df_prices.index <= df_w.index.values[-1])]



### Backtest Results

For the backtest, the training period for the HS starts in 2009-01-01, while the estimation of the returns and covariances begins in 2010-01-01 using data starting in 1995. 
Benchmark is the SP500.

'''python 

	def main():
	    _price_frequency = "W"
	    _scenario_frequency = "M"
	    _estimation_test_date = "2010-01-01"
	    _scenario_test_date = "2009-01-01"
	
	    model_estimation = ModelEstimation()
	    model_inputs_dict = model_estimation.ESTIMATION_MODEL(estimation_model = "Simple Average",
	                               estimation_test_date=_estimation_test_date,price_frequency=_price_frequency,scenario_frequency=_scenario_frequency,
	                               scenario_test_date=_scenario_test_date)
	
	    df_weights = model_estimation.PORT_OPTIMIZATION(model_inputs_dict,"MAX SHARPE")[0]  
	    df_comparison = model_estimation.backtest_performance(df_weights,_price_frequency)[0]
	    return df_comparison



### Accumulated Performance over time (Strategy vs. Benchmark) 
![image](https://github.com/user-attachments/assets/625c2d3f-e7a3-407d-9489-489cdde44c04)


### Performance Stats over time. 
      accumulated_return_strategy  accumulated_return_bench  difference
Year
2010                        12.44                     12.70       -0.26
**2011                         3.76                      0.68        3.08**
2012                        11.57                     10.84        0.73
2013                        31.47                     31.30        0.17
2014                        17.31                     13.43        3.88
**2015                         3.28                     -1.33        4.61**
2016                        11.23                      9.84        1.39
2017                        24.71                     18.10        6.61
**2018                        -4.52                     -7.03        2.50**
2019                        31.45                     30.34        1.10
2020                        16.07                     14.29        1.78
2021                        23.57                     27.62       -4.05
**2022                       -14.96                    -18.64        3.68**
2023                        22.26                     24.06       -1.80
2024                        22.34                     18.42        3.92


Overall Performance Comparison:

Strategy:
Accumulated Return: 5.5847
Volatility: 0.0746
Skew: -0.6182

Benchmark:
Accumulated Return: 4.0654
Volatility: 0.0780
Skew: -0.5433





