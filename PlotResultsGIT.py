
import numpy as np
import pandas as pd
import yfinance as yf
import BacktestScenarioEstimationGit
import BacktestModelEstimationGit

class PlotResults():
    def __init__(self):       
        self.hmm_model = BacktestScenarioEstimationGit.ScenarioEstimation()
        self.model_estimation = BacktestModelEstimationGit.ModelEstimation()
        df_price = pd.read_csv("PRICE_DATA.csv",index_col = 0)
        df_price.index = pd.to_datetime(df_price.index)
        self.df_price = df_price

        df_benchmark_price = yf.Ticker("^GSPC")
        df_benchmark_price = df_benchmark_price.history(period="max")
        df_benchmark_price.index = df_benchmark_price.index.date
        df_benchmark_price.index = pd.to_datetime(df_benchmark_price.index)
        self.df_benchmark_price =df_benchmark_price
        
       
    
    
    def plot_states_INDEX(self,scenario_frequency,scenario_test_date):
        import matplotlib.pyplot as plt
        dict_of_scenarios,df_scenarios,df_stats = self.hmm_model.HMMScenariosTrainTest(hmm_model="Macro",
                                                                           _frequency = scenario_frequency,
                                                                           test_date=scenario_test_date)
        
        df_price = self.df_benchmark_price.copy()
        df_price = df_price.resample(scenario_frequency).last()
        df_price = df_price.apply(np.log10)

        df = df_price.merge(df_scenarios,how = "inner",right_index = True, left_index = True)
        
        # Define the colors for each hidden state
        state_colors = {1: 'blue', 2: 'purple', 3: 'orange',4:"yellow"}

        # Create the plot
        plt.figure(figsize=(10, 6))

        # Plot the line for prices over time
        plt.plot(df.index, df[df.columns[0]], color='black', linestyle='-', linewidth=1, label='Price Line')

        # Plot scatter points for prices over time, colored by hidden state
        for state, color in state_colors.items():
            # Filter DataFrame by hidden state
            df_state = df[df['hidden_states'] == state]
            # Plot scatter points only
            plt.scatter(df_state.index, df_state[df_state.columns[0]], color=color, label=f'State {state}')

        plt.xlabel('Time')
        plt.ylabel('Prices')
        plt.title('Prices Over Time with Hidden States')
        plt.legend(title='Hidden States')
        print(f"The LAST HIDDEN STATE as of {df_scenarios.index.values[-1]}",df_scenarios.iloc[-1]["hidden_states"])
        plt.show()

    

    def plot_4D(self,scenario_frequency,scenario_test_date):
        import plotly.express as px
        # Creating a 3D scatter plot using Plotly
        import matplotlib.pyplot as plt
        dict_of_scenarios,df_scenarios,df_stats = self.hmm_model.HMMScenariosTrainTest(hmm_model="Macro",
                                                                           _frequency = scenario_frequency,
                                                                           test_date=scenario_test_date)
        fig = px.scatter_3d(df_scenarios, 
                            x='Real GDP', 
                            y='CPI', 
                            z='Market Returns', 
                            size='Market Volatility',
                            size_max=22,  # Size of points based on Market Volatility
                            color='hidden_states',     # Color based on Hidden States
                            hover_data=['Market Volatility'],  # Show Market Volatility on hover
                            title='3D Scatter Plot of Economic Indicators with Market Volatility')

        fig.update_traces(marker=dict(opacity=0.7, sizemode='area', line=dict(width=0.5, color='DarkSlateGrey')))

        print(f"The LAST HIDDEN STATE as of {df_scenarios.index.values[-1]}",df_scenarios.iloc[-1]["hidden_states"])
        fig.show()





def main():
   
    #Parameters: 
    _scenario_frequency = "M"

    _scenario_test_date = "2007-01-01"
    
    plot_results = PlotResults()
    plot_results.plot_states_INDEX(scenario_frequency = _scenario_frequency,scenario_test_date=_scenario_test_date)
    plot_results.plot_4D(scenario_frequency = _scenario_frequency,scenario_test_date=_scenario_test_date)


#main()



        