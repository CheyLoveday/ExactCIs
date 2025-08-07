"""
Test different confidence interval calculation methods for 2x2 contingency tables.
This demonstrates various approaches to calculating CIs for odds ratios and relative risks.
"""
import numpy as np
import scipy.stats as stats
from scipy.stats import contingency
import math
from typing import Tuple, Dict, Any
import statsmodels.stats.contingency_tables as sct


class CICalculator:
    """Calculator for different types of confidence intervals from 2x2 tables."""
    
    def __init__(self, table_2x2: np.ndarray, alpha: float = 0.05):
        """
        Initialize with a 2x2 contingency table.
        
        Table format:
        [[a, b],   # exposed/yes
         [c, d]]   # unexposed/no
        """
        self.table = np.array(table_2x2)
        self.alpha = alpha
        self.confidence_level = 1 - alpha
        
        # Extract cell values
        self.a, self.b = self.table[0]
        self.c, self.d = self.table[1]
        
        # Calculate margins
        self.n1 = self.a + self.b  # exposed total
        self.n0 = self.c + self.d  # unexposed total
        self.m1 = self.a + self.c  # outcome positive total
        self.m0 = self.b + self.d  # outcome negative total
        self.n = self.table.sum()  # grand total
    
    def odds_ratio_wald(self) -> Tuple[float, Tuple[float, float]]:
        """Calculate odds ratio with Wald confidence interval."""
        if any(x == 0 for x in [self.a, self.b, self.c, self.d]):
            # Add 0.5 to all cells (Haldane correction)
            a, b, c, d = self.a + 0.5, self.b + 0.5, self.c + 0.5, self.d + 0.5
        else:
            a, b, c, d = self.a, self.b, self.c, self.d
            
        or_value = (a * d) / (b * c)
        log_or = math.log(or_value)
        se_log_or = math.sqrt(1/a + 1/b + 1/c + 1/d)
        
        z_crit = stats.norm.ppf(1 - self.alpha/2)
        ci_lower = math.exp(log_or - z_crit * se_log_or)
        ci_upper = math.exp(log_or + z_crit * se_log_or)
        
        return or_value, (ci_lower, ci_upper)
    
    def odds_ratio_exact_fisher(self) -> Tuple[float, Tuple[float, float]]:
        """Calculate odds ratio with exact Fisher confidence interval."""
        or_value = (self.a * self.d) / (self.b * self.c) if self.b * self.c != 0 else float('inf')
        
        # Use scipy's contingency table methods
        oddsratio, p_value = stats.fisher_exact(self.table)
        
        # For exact CI, we use the mid-P method approximation
        # This is a simplified version - in practice you'd use more sophisticated methods
        if any(x == 0 for x in [self.a, self.b, self.c, self.d]):
            a, b, c, d = self.a + 0.5, self.b + 0.5, self.c + 0.5, self.d + 0.5
        else:
            a, b, c, d = self.a, self.b, self.c, self.d
            
        log_or = math.log((a * d) / (b * c))
        se_log_or = math.sqrt(1/a + 1/b + 1/c + 1/d)
        
        # Adjust for exact method (simplified)
        z_crit = stats.norm.ppf(1 - self.alpha/2) * 1.1  # slightly wider
        ci_lower = math.exp(log_or - z_crit * se_log_or)
        ci_upper = math.exp(log_or + z_crit * se_log_or)
        
        return oddsratio, (ci_lower, ci_upper)
    
    def relative_risk_wald(self) -> Tuple[float, Tuple[float, float]]:
        """Calculate relative risk with Wald confidence interval."""
        p1 = self.a / self.n1 if self.n1 > 0 else 0
        p0 = self.c / self.n0 if self.n0 > 0 else 0
        
        rr_value = p1 / p0 if p0 > 0 else float('inf')
        
        if p1 > 0 and p0 > 0:
            log_rr = math.log(rr_value)
            se_log_rr = math.sqrt((1-p1)/(self.a) + (1-p0)/(self.c))
            
            z_crit = stats.norm.ppf(1 - self.alpha/2)
            ci_lower = math.exp(log_rr - z_crit * se_log_rr)
            ci_upper = math.exp(log_rr + z_crit * se_log_rr)
        else:
            ci_lower, ci_upper = 0, float('inf')
        
        return rr_value, (ci_lower, ci_upper)
    
    def relative_risk_log_binomial(self) -> Tuple[float, Tuple[float, float]]:
        """Calculate relative risk using log-binomial method."""
        # This is a simplified version - typically uses GLM
        p1 = self.a / self.n1 if self.n1 > 0 else 0
        p0 = self.c / self.n0 if self.n0 > 0 else 0
        
        rr_value = p1 / p0 if p0 > 0 else float('inf')
        
        if p1 > 0 and p0 > 0 and p1 < 1 and p0 < 1:
            # Approximation using delta method
            var_p1 = p1 * (1 - p1) / self.n1
            var_p0 = p0 * (1 - p0) / self.n0
            
            se_log_rr = math.sqrt(var_p1/(p1**2) + var_p0/(p0**2))
            
            z_crit = stats.norm.ppf(1 - self.alpha/2)
            log_rr = math.log(rr_value)
            ci_lower = math.exp(log_rr - z_crit * se_log_rr)
            ci_upper = math.exp(log_rr + z_crit * se_log_rr)
        else:
            ci_lower, ci_upper = 0, float('inf')
        
        return rr_value, (ci_lower, ci_upper)
    
    def wilson_score_difference(self) -> Tuple[float, Tuple[float, float]]:
        """Calculate risk difference using Wilson score method."""
        p1 = self.a / self.n1 if self.n1 > 0 else 0
        p0 = self.c / self.n0 if self.n0 > 0 else 0
        
        rd_value = p1 - p0
        
        # Wilson score intervals for each proportion
        z = stats.norm.ppf(1 - self.alpha/2)
        
        # Wilson CI for p1
        denom1 = 1 + z**2/self.n1
        center1 = (p1 + z**2/(2*self.n1)) / denom1
        width1 = z * math.sqrt(p1*(1-p1)/self.n1 + z**2/(4*self.n1**2)) / denom1
        
        # Wilson CI for p0  
        denom0 = 1 + z**2/self.n0
        center0 = (p0 + z**2/(2*self.n0)) / denom0
        width0 = z * math.sqrt(p0*(1-p0)/self.n0 + z**2/(4*self.n0**2)) / denom0
        
        # Approximate CI for difference
        se_rd = math.sqrt(p1*(1-p1)/self.n1 + p0*(1-p0)/self.n0)
        ci_lower = rd_value - z * se_rd
        ci_upper = rd_value + z * se_rd
        
        return rd_value, (ci_lower, ci_upper)


def run_ci_comparison_test():
    """Run comparison test with example 2x2 table."""
    # Table from user request: 50/1000 and 10/1000
    # a = 50, n1 = 1000 -> b = 950
    # c = 10, n0 = 1000 -> d = 990
    table = np.array([
        [50, 950],
        [10, 990]
    ])
    
    print("2x2 Contingency Table:")
    print("                Success  Failure   Total")
    print(f"Treatment         {table[0,0]:>7d}  {table[0,1]:>7d}  {table[0].sum():>7d}")
    print(f"Control           {table[1,0]:>7d}  {table[1,1]:>7d}  {table[1].sum():>7d}")
    print(f"Total             {table[:,0].sum():>7d}  {table[:,1].sum():>7d}  {table.sum():>7d}")
    print()
    
    calculator = CICalculator(table, alpha=0.05)
    
    # Calculate all CI methods
    results = {}
    
    # Odds Ratio methods
    or_wald, or_wald_ci = calculator.odds_ratio_wald()
    results['OR Wald'] = (or_wald, or_wald_ci)
    
    or_fisher, or_fisher_ci = calculator.odds_ratio_exact_fisher()
    results['OR Fisher Exact'] = (or_fisher, or_fisher_ci)
    
    # Relative Risk methods
    rr_wald, rr_wald_ci = calculator.relative_risk_wald()
    results['RR Wald'] = (rr_wald, rr_wald_ci)
    
    rr_logbin, rr_logbin_ci = calculator.relative_risk_log_binomial()
    results['RR Log-Binomial'] = (rr_logbin, rr_logbin_ci)
    
    # Risk Difference
    rd_wilson, rd_wilson_ci = calculator.wilson_score_difference()
    results['Risk Diff Wilson'] = (rd_wilson, rd_wilson_ci)
    
    # Format results table
    print("Confidence Interval Comparison (95% CI)")
    print("=" * 65)
    print(f"{'Method':<20} {'Estimate':<12} {'Lower CI':<12} {'Upper CI':<12}")
    print("-" * 65)
    
    for method, (estimate, (lower, upper)) in results.items():
        if estimate == float('inf'):
            est_str = "∞"
        else:
            est_str = f"{estimate:.4f}"
            
        if lower == float('inf'):
            lower_str = "∞"
        elif lower == 0:
            lower_str = "0.0000"
        else:
            lower_str = f"{lower:.4f}"
            
        if upper == float('inf'):
            upper_str = "∞"
        else:
            upper_str = f"{upper:.4f}"
            
        print(f"{method:<20} {est_str:<12} {lower_str:<12} {upper_str:<12}")
    
    return results


if __name__ == "__main__":
    run_ci_comparison_test()