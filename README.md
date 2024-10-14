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

Through a simple combination of simply 4 features, the project explores the use of a Hidden Markov Model in defining a set of hidden states (market regimes). 
Each hidden state is determined using as inputs the the VIX, market returns (SP500), CPI, and industrial production index (INDPRO). These variables help identify and classify the underlying market conditions that the hidden state captures. In other words, each hidden state encapsulates a historical phase where these variables exhibited similar behaviors, allowing for the classification of market conditions into regimes with shared structural characteristics.



