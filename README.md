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
![image](https://github.com/user-attachments/assets/2643e436-15b5-4cdf-8079-781f903c3e60)

We can observe that Hidden State (HS) 1 is characteristic of periods of higher inflation while the remaining dots are more of low to normal volatility. 
Furthermore HS 2 is characterized of periods of lower volatility and an average market return, suggesting moments of trending bull markets. 
HS 4 is characterized by periods of lower returns and also lower INDPRO, associated with periods of economic contraction 
Finally, HS 3 is characterized by the periods with the highest level of the VIX. 

Hidden States	INDPRO		CPI		Market Returns		Market Volatility	
	count	mean	count	mean	count	mean	count	mean
1	49	0.0%	49	5.8%	49	0.6%	49	20.9
2	191	0.2%	191	2.4%	191	1.0%	191	14.1
3	139	0.2%	139	2.5%	139	0.6%	139	22.8
4	34	-0.5%	34	1.2%	34	0.1%	34	34.8
![image](https://github.com/user-attachments/assets/dc948cb1-896e-46bb-96f2-51fe83be6c85)

Furthermore we can inspect how these HS's relate with market performance over time, by plotting theem in relation to the SP500 performance. 

![hidden states](https://github.com/user-attachments/assets/c4b52539-721d-40a9-a552-949981a0eaea)

As expected, HS 2 periods are typical of trending markets. HS 3 associated with periods of higher volatility, with a positive or negative direction. HS 1, related with a higher level of inflation is characterized of periods of lower market performance.



# Portfolio Optimization

# Backtest Results



