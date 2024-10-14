
import cvxpy as cp
import numpy as np

class CustomOptimizer:

    def optimize_MVP(self, RET, COV, risk_free_rate, max_weight, min_weight):
        n = len(RET)
        w = cp.Variable(n)
        portfolio_variance = cp.quad_form(w, COV)
        
        objective = cp.Minimize(portfolio_variance)
        
        constraints = [
            cp.sum(w) == 1,  # Sum of weights equals 1
            w >= 0,          # Long-only constraint
        ]
        
        if max_weight is not None:
            constraints += [w <= max_weight]
        
        if min_weight is not None:
            constraints += [w >= min_weight]
        
        problem = cp.Problem(objective, constraints)
        
        problem.solve(solver=cp.ECOS)
        

        
        return w.value
        

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
    

    def solve_sharpe_ratio_LS(self, RET, COV, risk_free_rate, risk_target_value, lambda_reg, max_weight,min_weight):
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
            #w >= 0,          # Long-only constraint
            portfolio_variance <= risk_target_value  # Variance constraint
        ]
        
        # Add maximum weight constraint if provided
        if max_weight is not None:
            constraints += [w <= max_weight]

        if min_weight is not None:
            constraints += [w >= min_weight]
        
        # Define the problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS, qcp=True)  # Indicate that the problem is a QCP
        
        return w.value, portfolio_return.value, portfolio_variance.value
    

    
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
        
        return best_weights#, best_sharpe_ratio

    def optimize_risk_parity(self, COV, lambda_reg=None, max_weight=None):
        # Ensure the covariance matrix is a NumPy array
        cov_matrix = np.asarray(COV)
        
        # Compute the inverse of the covariance matrix
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        
        # Compute the diagonal of the inverse covariance matrix
        diag_inv_cov = np.diag(inv_cov_matrix)        
        # Compute the risk contribution for each asset
        risk_contributions = diag_inv_cov / np.sum(diag_inv_cov)
    
        # Compute the risk parity weights
        weights = risk_contributions / np.sum(risk_contributions)
        best_weights = weights
        return best_weights
    

    def optimize_heuristic_approach(self, RET):

        # Min-Max Normalization
        min_val = np.min(RET)
        max_val = np.max(RET)

        normalized_values = (RET - min_val) / (max_val - min_val)

        best_weights = normalized_values / normalized_values.sum()

        # The normalized values can be used as weights
        return best_weights

