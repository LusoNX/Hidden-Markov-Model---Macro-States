
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import os 

class ScenarioEstimation():
    
    # -------------------------------------------------------------------- HELPER FUNCTIONS ------------------------------------------------------------------------------------
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
            #bic = model.bic(observed_data)
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
    

    # Run several simulations. 
    def transform_matrix(self,df):
        # Create a copy of the dataframe to avoid modifying the original one
        df_transformed = df.copy()
        
        # Iterate over each column
        for col in df.columns:
            # Get the unique values and their order of appearance
            unique_values = pd.Series(df[col]).unique()
            value_to_label = {val: idx + 1 for idx, val in enumerate(unique_values)}
            
            # Map the original values to their new labels
            df_transformed[col] = df[col].map(value_to_label)
            
        return df_transformed,value_to_label


    def MacroVariables(self,_frequency):
        # Import the input features for the estimation 
        # Note. You can automatize the import through Fred API (Fed API) for 3 of the 4 variables used (VIX, INDPRO and CPI) 


        df_INDPRO = pd.read_csv(rf"{os.getcwd()}\DATA\INDPRO.csv",index_col =0)
        df_INDPRO = df_INDPRO.rename(columns = {"Value":"Real GDP"})
        df_INDPRO.index = pd.to_datetime(df_INDPRO.index)

        df_CPI = pd.read_csv(rf"{os.getcwd()}\DATA\CPI.csv",index_col =0)
        df_CPI = df_CPI.rename(columns = {"Value":"CPI"})
        df_CPI.index = pd.to_datetime(df_CPI.index)

        df_VIX = pd.read_csv(rf"{os.getcwd()}\DATA\VIX.csv",index_col =0)
        df_VIX = df_VIX.rename(columns = {"Value":"Market Volatility"})
        df_VIX.index = pd.to_datetime(df_VIX.index)

        # Uses yfinance 
        df_sp500 = yf.Ticker("^GSPC")
        df_sp500 = df_sp500.history(period="max")
        df_sp500.index = df_sp500.index.date

        df_sp500.index = pd.to_datetime(df_sp500.index)
        df_sp500 = df_sp500[["Close"]]
        df_sp500 = df_sp500.resample(_frequency).last()

    
        # Set Frequencies
        if _frequency == "M":
            df_INDPRO =df_INDPRO.pct_change(1).iloc[1::]
            df_INDPRO.index = pd.to_datetime(df_INDPRO.index) + pd.offsets.MonthEnd(0)

            df_CPI =df_CPI.pct_change(12).iloc[12::]
            df_CPI.index = pd.to_datetime(df_CPI.index) + pd.offsets.MonthEnd(0)

            df_sp500["Market Returns"] =df_sp500["Close"].pct_change(1).iloc[1::]
            df_sp500 = df_sp500.resample(_frequency).last()

            df_VIX.index = pd.to_datetime(df_VIX.index)
            df_VIX = df_VIX.resample(_frequency).last()

            

        elif _frequency == "W":
            df_INDPRO =df_INDPRO.pct_change(1).iloc[1::]
            df_INDPRO.index = pd.to_datetime(df_INDPRO.index) + pd.offsets.MonthEnd(0)
            df_INDPRO = df_INDPRO.asfreq(_frequency).ffill()

            df_CPI =df_CPI.pct_change(12).iloc[12::]
            df_CPI.index = pd.to_datetime(df_CPI.index) + pd.offsets.MonthEnd(0)
            df_CPI = df_CPI.asfreq(_frequency).ffill()

            df_sp500["Market Returns"] =df_sp500["Close"].pct_change(1).iloc[1::]
        
        else:
            print("Please provide a approppriate frequency : Q or M ")

        # DELETE THIS LATER 
        # last_row = df_INDPRO.iloc[[-1]].copy()
        # last_date = last_row.index
        # next_month_date =last_date + pd.DateOffset(months=1)
        # last_row.index= next_month_date
        # df_INDPRO = pd.concat([df_INDPRO,last_row])

        # # ##
        # last_row = df_CPI.iloc[[-1]].copy()
        # last_date = last_row.index
        # next_month_date =last_date + pd.DateOffset(months=1)
        # last_row.index= next_month_date
        # df_CPI = pd.concat([df_CPI,last_row])
        
        df = df_INDPRO.merge(df_CPI, how = "inner",right_index=True,left_index=True)
        df = df.merge(df_sp500, how = "inner",right_index=True,left_index=True)
        df = df.merge(df_VIX, how = "inner",right_index=True,left_index=True)

        df = df.sort_index()
        df = df[["Real GDP","CPI","Market Returns","Market Volatility"]]


        # Shift is required to prevent look-ahead bias.
        # We shift by 2, since the data is published only on the beggining of the second half of the next month. 
        # For example. 
        # CPI for July, was reported in 14 August. 
        # Since the data retrieved is referring to July data but only available after 14 August, 
        #   we must ensure that we are not incurring look ahead bias in July measurement of the scenario 
        df[["Real GDP","CPI"]]  = df[["Real GDP","CPI"]].shift(2)


        # For Market Volatility and returns we can simply shift by 1. 
        df[["Market Returns","Market Volatility"]] = df[["Market Returns","Market Volatility"]].shift(1)
        df = df.iloc[2::]


        return df

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
        

        def extract_from_describe(df_stats, *stats):
            return df_stats.loc[:, (slice(None), stats)]

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
                model = hmm.GaussianHMM(n_components=optimal_scenarios[hmm_model], covariance_type="full",params="stmcm",init_params="",n_iter=10000)#,random_state=59)#,random_state = 58)                
                model.startprob_  = prev_startprob
                model.transmat_ = prev_transmat
                model.means_ = prev_means

                try:
                    model.covars_ = prev_covar
                except:
                    # Ensures that the covariances are symmetric and positive definite. 
                    for i in range(prev_covar.shape[0]):
                        epsilon = 1e-6
                        prev_covar[i] = (prev_covar[i] + prev_covar[i].T) / 2
                        prev_covar[i] += np.eye(prev_covar[i].shape[0]) * epsilon

                    model.covars_ = prev_covar
                
            
            try:
                # In case extreme outliers after scaling emerge,  clip values to a min/max zscores of -(+) 3
                scaled_values = np.clip(scaled_values, -3, 3)
                model.fit(scaled_values)
                prev_startprob = model.startprob_
                prev_transmat = model.transmat_
                prev_means = model.means_
                prev_covar = model.covars_

                hidden_states = model.predict(scaled_values)
                prev_model = model
            except:
                model = prev_model
                hidden_states = model.predict(scaled_values)
                nr_of_scenario_failures +=1 

            #Transforms the matrix automatically, so the scenarios are ALLWAYS ORDERED IN THE CORRECT FORMAT. 
            # This is useful when testing whether the model scenarios result in disctint results in each run. If yes, then the model is not able to converge to a proper solution. 
            df_new["hidden_states"] = hidden_states
            df_transformed = self.transform_matrix(df_new[["hidden_states"]])[0]
            df_new["hidden_states"] = df_transformed["hidden_states"]
        
            df_new.index.name = "Date"
            dict_of_inputs[date_str] = {"SCENARIO_DF":df_new,"Hidden State":hidden_states[-1]}
            df_groupby_stats = df_new.groupby('hidden_states').describe()
            specific_stats = extract_from_describe(df_groupby_stats, 'count', 'mean')


            if i ==0:
                df_scenarios_store = df_new
            else:
                df_scenarios_store = pd.concat([df_scenarios_store,df_new.iloc[[-1]]])
            
        
        print(f"Nr of Failed Scenarios: {nr_of_scenario_failures} in total of {len(test_df)}")
        return dict_of_inputs,df_scenarios_store,specific_stats
    

    
    def MC_HMMScenarios(self,hmm_model,_frequency,test_date,is_update):
       ## Function used to run several model simulations, with different random seeds in order to evaluate the robustness of the model.
       # If there is no pairwise agreement among the models, the model is not robust and fails to converge to a common solution  

        def calculate_pairwise_agreement(df_transformed):
            from sklearn.metrics import pairwise_distances
            matrix = df_transformed.values
            
            # Number of columns
            n_cols = matrix.shape[1]
            agreement_scores = []
            
            for i in range(n_cols):
                for j in range(i + 1, n_cols):
                    # Calculate agreement (proportion of identical labels)
                    agreement = np.mean(matrix[:, i] == matrix[:, j])
                    agreement_scores.append(agreement)
            
            average_agreement = np.mean(agreement_scores)
            return average_agreement
    
        df_list = []
        all_means = {}
        df_stats = []

        for x in range(10):
            dict_1,df_scenarios,specific_stats = self.HMMScenariosTrainTest(hmm_model,_frequency,test_date,is_update)
            specific_stats["ITERATION"] = x
            df_grouped = df_scenarios.groupby('hidden_states').mean()
            all_means[f'Run_{x}'] = df_grouped[['Real GDP', 'CPI', 'Market Returns', 'Market Volatility']].reset_index()

            df_transformed,value_to_label =self.transform_matrix(df_scenarios[["hidden_states"]])
            df_list.append(df_transformed)

            specific_stats.index = specific_stats.index.map(value_to_label)
            specific_stats = specific_stats.sort_index()

            df_stats.append(specific_stats)
   
        #means_dfs = [all_means[key] for key in sorted(all_means.keys())]
        df_stats = pd.concat(df_stats)
        #df_stats.to_excel("STATS.xlsx")

        df = pd.concat(df_list,axis = 1)
        #df.to_excel("SCENARIOS_NORMAL.xlsx")


def main():
    hmm_model = ScenarioEstimation()
    hmm_model.HMMScenariosTrainTest(hmm_model="Macro",_frequency = "M",test_date="2006-01-01",is_update=True)
    hmm_model.MC_HMMScenarios(hmm_model="Macro",_frequency = "M",test_date="2010-01-01",is_update=True)
#main()


