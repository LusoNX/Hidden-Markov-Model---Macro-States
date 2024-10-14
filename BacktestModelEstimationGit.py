
from functools import reduce
import numpy as np
import pandas as pd
import sys
import yfinance as yf
import sys
import BacktestScenarioEstimationGit
import CustomOptimizer
import os 



# This script is simply used to UNDERSTAND WHERE WE STAND ON THE DISTRIBUTION OF THE GDP, to know which portfolio is more adequate and calculate a mix between the optimal 3 (Good,Normal and Bad)
class ModelEstimation():
    def __init__(self):        
        self.hmm_model = BacktestScenarioEstimationGit.ScenarioEstimation()
        self.id_to_name_dict = {22:"Value",23:"Growth",24:"Momentum",25:"Low Volatility"}
        self.cur_dir = os.getcwd()
    
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
            filtered_correlation_matrix = (eigenvectors @ np.diag(filtered_eigenvalues) @ eigenvectors.T)
            filtered_correlation_matrix[np.diag_indices_from(filtered_correlation_matrix)] = 1
            volatilities = np.std(df, axis=0)

            # Rescale the filtered correlation matrix by the volatilities
            filtered_covariance_matrix = filtered_correlation_matrix * np.outer(volatilities, volatilities)  
            port_cov = pd.DataFrame(filtered_covariance_matrix, index=list(df.columns), columns=list(df.columns))
                                
        return port_cov
    


    def ESTIMATION_MODEL(self,estimation_model,estimation_test_date,price_frequency,scenario_frequency,scenario_test_date):
    
        scenario_test_date_adj = pd.to_datetime(scenario_test_date)
        estimation_test_date_adj = pd.to_datetime(estimation_test_date)
        # The date for the backtest estimation model must exceed the one set for the scenario determination. 
        if estimation_test_date_adj < scenario_test_date_adj:
            sys.exit("Estimation test date must be after Scenario test date")
        
        dict_of_scenarios,df_scenarios_store,df_stats = self.hmm_model.HMMScenariosTrainTest(hmm_model="Macro",_frequency = scenario_frequency,test_date=scenario_test_date)
        df_price = pd.read_csv(rf"{self.cur_dir}\DATA\PRICE_DATA.csv",index_col = 0)
        df_price.index = pd.to_datetime(df_price.index) 
        df_price_D = df_price.copy()
        df_price_D = df_price_D[df_price_D.index.weekday < 5]
        df_price_D = df_price_D.pct_change().iloc[1::]

        df_price = df_price.resample(price_frequency).last()
        df_price = df_price.pct_change().iloc[1::]

        if price_frequency == "D":
            df_price = df_price[df_price.index.weekday < 5]

        df_test = df_price[df_price.index >=estimation_test_date_adj]

    
        test_window = len(df_test)
        model_inputs_dict = {}


        for i in range(test_window):
            _date = df_test.iloc[[i]].index.values[0]
            df_price_ind = df_price[df_price.index <= _date]
            df_scenarios = df_scenarios_store[df_scenarios_store.index <= _date]
            hidden_state = int(df_scenarios.iloc[-1]["hidden_states"])

            #USED FOR THE DAILY COVARIANCE
            df_scenarios_adj = df_scenarios.asfreq("D", method='ffill')
            df_scenarios_adj = df_scenarios_adj[df_scenarios_adj.index <= _date]
            df_scenarios_adj = df_scenarios_adj[df_scenarios_adj["hidden_states"]==df_scenarios_adj["hidden_states"].iloc[-1]]
            covariance_index = list(df_scenarios_adj.index.values)

            df_scenarios = df_scenarios.asfreq(price_frequency, method='ffill')
            df_merge =df_price_ind.merge(df_scenarios[["hidden_states"]],how ="inner",right_index = True, left_index = True)
            

            # In moments of month change, this is required to ensure we are using all available data up to the moment of the estimation of the weights. 
            if price_frequency == "W" or price_frequency =="D":
                df_concat = df_price_ind.iloc[[-1]]
                df_concat["hidden_states"] = hidden_state
                #print(df_merge)
                df_merge = pd.concat([df_merge,df_concat])
            

            if estimation_model == "Simple Average":
                
                df_merge = df_merge[df_merge["hidden_states"]==df_merge["hidden_states"].iloc[-1]]
                if price_frequency == "D":
                    df_merge = df_merge[df_merge.index.weekday <5]

                df_merge.iloc[:,:-1] = df_merge.iloc[:,:-1].ewm(span=3, adjust=False).mean()
                   
                RET = df_merge.iloc[-1:,:-1].iloc[-1]


                df_price_D_ind = df_price_D[df_price_D.index<=_date]
                df_price_D_ind = df_price_D_ind[df_price_D_ind.index.isin(covariance_index)]
                COV = self.helper_covariance_parameters(df_price_D_ind,"random_matrix","D")
                    

          
            model_inputs_dict[_date] = {"RET":RET,"COV":COV,"Hidden State":hidden_state}
            self.df_scenario_stats = df_stats
        
        return model_inputs_dict



    def PORT_OPTIMIZATION(self,model_inputs_dict,optimization_type):
        ef = CustomOptimizer.CustomOptimizer()

        weights_list = []
        weight_failures = 0
        dates_failure = []
        data_values = []
        
        for date in model_inputs_dict.keys():
            port_ret = model_inputs_dict[date]["RET"]      
            port_cov = model_inputs_dict[date]["COV"]
            hidden_states= model_inputs_dict[date]["Hidden State"]
            data_values.append([date,port_ret,hidden_states])            
            max_vol_state = self.df_scenario_stats["Market Volatility"]["mean"].idxmax()

            try:
                if optimization_type == "MAX SHARPE":
                    best_weights = ef.optimize_sharpe_ratio(RET = port_ret.values, COV=port_cov.values, risk_free_rate=0, lambda_reg=0.01,max_weight=None,min_weight=None,long_short=False)
                elif optimization_type == "HEURISTIC":
                    best_weights = ef.optimize_heuristic_approach(RET = port_ret.values)
    

            except:
                best_weights = None
            
            if best_weights is None:
                best_weights = weights_list[-1]
                best_weights = best_weights.drop(columns="Date")
                best_weights = best_weights.values[0]
                weight_failures+=1
                dates_failure.append(date)
            selected_ids = list(port_cov.columns)
            df_weights = pd.DataFrame([best_weights],columns = selected_ids)
            df_weights["Date"]  = date
            weights_list.append(df_weights)


        
        df_W = pd.concat(weights_list)
        df_W = df_W.set_index("Date")
        df = pd.DataFrame(data_values,columns=["Date","Returns","Hidden State"])
        print(f"Nrº of Weight failures:{weight_failures}")
        print(f"Dates failure:{dates_failure}")

        df_W_adj =df_W.rename(columns = self.id_to_name_dict)
        df_W_adj["HHI_CONCENTRATION"] = df_W_adj.apply(lambda row:(row**2).sum(),axis=1)

        df_W_adj.to_excel("WEIGHTS_OVERTIME.xlsx")
        
        return df_W,weight_failures
    

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

        df_multiplied = df_prices * df_w

        _ids = list(df_multiplied.columns)
        df_multiplied["SUM"] = df_multiplied.sum(axis=1)
        df_multiplied["Cum Prod (STRAT)"] = (1+df_multiplied["SUM"]).cumprod()

        df_best_class = df_multiplied[_ids].copy()
        df_best_class =df_best_class.rename(columns = self.id_to_name_dict)
        df_best_class['Best Class'] = df_best_class.idxmax(axis=1)
        
        df_benchmark_prices = yf.Ticker("^GSPC")
        df_benchmark_prices = df_benchmark_prices.history(period="max")
        df_benchmark_prices.index = df_benchmark_prices.index.date
        df_benchmark_prices.index = pd.to_datetime(df_benchmark_prices.index)
        df_benchmark_prices = df_benchmark_prices[["Close"]]
        df_benchmark_prices = df_benchmark_prices.resample(_frequency).last()
        df_benchmark_prices = df_benchmark_prices.pct_change().iloc[1::]

        df_benchmark_prices = df_benchmark_prices[(df_benchmark_prices.index >= df_multiplied.index.values[0]) & (df_benchmark_prices.index <= df_multiplied.index.values[-1])]
        df_benchmark_prices["Cum Prod (BENCH)"] = (1+df_benchmark_prices["Close"]).cumprod()
        df_merge = df_benchmark_prices[["Cum Prod (BENCH)"]].merge(df_multiplied[["Cum Prod (STRAT)"]],how = "inner",right_index=True,left_index=True)

        def performance_outlook(DF, DF_BENCH, _column, _column_bench):
            from scipy.stats import skew
            df = DF.copy()
            df_bench = DF_BENCH.copy()

            df_perf = df[[_column]]
            df_bench = df_bench[[_column_bench]]
            
            # Extract year from the index
            df_perf['Year'] = df_perf.index.year
            df_bench['Year'] = df_bench.index.year

            # Group by year and calculate accumulated return for each year
            yearly_performance_strategy = df_perf.groupby('Year').agg(
                accumulated_return=pd.NamedAgg(column=_column, aggfunc=lambda x: (1 + x).prod() - 1)
            )
            
            yearly_performance_bench = df_bench.groupby('Year').agg(
                accumulated_return=pd.NamedAgg(column=_column_bench, aggfunc=lambda x: (1 + x).prod() - 1)
            )
            
            yearly_comparison = yearly_performance_strategy.join(yearly_performance_bench, lsuffix='_strategy', rsuffix='_bench')
            yearly_comparison['difference'] = yearly_comparison['accumulated_return_strategy'] - yearly_comparison['accumulated_return_bench']
            yearly_comparison = yearly_comparison*100
            yearly_comparison = yearly_comparison.round(2)
            print("Yearly Performance Comparison:")
            print(yearly_comparison)
            
            # Overall accumulated return, volatility, skew for strategy
            overall_accumulated_return_strategy = (1 + df_perf[_column]).prod() - 1
            overall_volatility_strategy = df_perf[_column].std() * np.sqrt(12)
            overall_skewness_strategy = skew(df_perf[_column])
            
            # Overall accumulated return, volatility, skew for benchmark
            overall_accumulated_return_bench = (1 + df_bench[_column_bench]).prod() - 1
            overall_volatility_bench = df_bench[_column_bench].std() * np.sqrt(12)
            overall_skewness_bench = skew(df_bench[_column_bench])
            

            print("\nOverall Performance Comparison:")
            print("Strategy:")
            print(f"Accumulated Return: {overall_accumulated_return_strategy:.4f}")
            print(f"Volatility: {overall_volatility_strategy:.4f}")
            print(f"Skew: {overall_skewness_strategy:.4f}")
            
            print("\nBenchmark:")
            print(f"Accumulated Return: {overall_accumulated_return_bench:.4f}")
            print(f"Volatility: {overall_volatility_bench:.4f}")
            print(f"Skew: {overall_skewness_bench:.4f}")
            return yearly_comparison[["difference"]],overall_volatility_strategy, \
                    overall_volatility_bench,overall_skewness_strategy, \
                    overall_skewness_bench
        
        df_comparison,overall_volatility_strategy, \
        overall_volatility_bench,overall_skewness_strategy, \
        overall_skewness_bench = performance_outlook(df_multiplied, df_benchmark_prices, 'SUM',"Close")
        df_merge.plot()
        plt.show()


        items_data = [overall_volatility_strategy,overall_volatility_bench,overall_skewness_strategy,overall_skewness_bench]
        selected_ids = list(df_prices.columns)
        return df_comparison,items_data,selected_ids
        

            
def main():
    _price_frequency = "D"
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

main()

