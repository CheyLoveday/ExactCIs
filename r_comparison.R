#!/usr/bin/env Rscript

# Load required libraries
library(exact2x2)
library(jsonlite)

# Test cases matching our Python tests
test_cases <- list(
  # Standard tables
  list(a=7, b=3, c=2, d=8, name="Standard table 1"),
  list(a=10, b=10, c=10, d=10, name="Balanced small table"),
  
  # Tables with zero cells
  list(a=0, b=10, c=5, d=15, name="Table with a zero"),
  list(a=10, b=0, c=5, d=15, name="Table with b zero"),
  
  # Various table sizes and distributions
  list(a=40, b=10, c=20, d=30, name="Unbalanced medium table"),
  list(a=100, b=50, c=60, d=120, name="Large table 1"),
  list(a=500, b=500, c=300, d=700, name="Large table 2"),
  
  # Extreme proportions
  list(a=99, b=1, c=50, d=50, name="Extreme proportion 1"),
  list(a=1, b=99, c=50, d=50, name="Extreme proportion 2"),
  
  # Different group sizes
  list(a=5, b=5, c=90, d=10, name="Very different group sizes 1"),
  list(a=90, b=10, c=5, d=5, name="Very different group sizes 2")
)

# Alpha values to test
alpha_values <- c(0.05, 0.01, 0.1)

# Functions to calculate CIs with various methods
calculate_barnard_ci <- function(a, b, c, d, alpha = 0.05) {
  tryCatch({
    # Use Barnard's exact test from the exact2x2 package
    result <- exact2x2::barnard.test(matrix(c(a, b, c, d), nrow=2), 
                                     alternative="two.sided", 
                                     conf.level=1-alpha)
    
    ci <- result$conf.int
    return(list(lower=ci[1], upper=ci[2]))
  }, error = function(e) {
    cat("Error in Barnard's test for", a, b, c, d, ":", e$message, "\n")
    return(list(lower=NA, upper=NA))
  })
}

calculate_fisher_ci <- function(a, b, c, d, alpha = 0.05) {
  tryCatch({
    # Use Fisher's exact test
    result <- fisher.test(matrix(c(a, b, c, d), nrow=2), 
                          alternative="two.sided", 
                          conf.level=1-alpha)
    
    ci <- result$conf.int
    return(list(lower=ci[1], upper=ci[2]))
  }, error = function(e) {
    cat("Error in Fisher's test for", a, b, c, d, ":", e$message, "\n")
    return(list(lower=NA, upper=NA))
  })
}

calculate_exact_ci <- function(a, b, c, d, alpha = 0.05) {
  tryCatch({
    # Use unconditional exact test from exact2x2
    result <- exact2x2::exact2x2(matrix(c(a, b, c, d), nrow=2), 
                                  tsmethod="central", 
                                  conf.level=1-alpha)
    
    ci <- result$conf.int
    return(list(lower=ci[1], upper=ci[2]))
  }, error = function(e) {
    cat("Error in Unconditional Exact test for", a, b, c, d, ":", e$message, "\n")
    return(list(lower=NA, upper=NA))
  })
}

# Function to run all tests
run_all_tests <- function() {
  results <- list()
  
  for (alpha in alpha_values) {
    alpha_results <- list()
    
    for (test_case in test_cases) {
      a <- test_case$a
      b <- test_case$b
      c <- test_case$c
      d <- test_case$d
      name <- test_case$name
      
      if (alpha == 0.05) {
        test_name <- name
      } else {
        test_name <- paste0(name, " with alpha=", alpha)
      }
      
      cat("Testing", test_name, "...\n")
      
      # Get results from all methods
      start_time <- Sys.time()
      barnard_ci <- calculate_barnard_ci(a, b, c, d, alpha)
      barnard_time <- as.numeric(difftime(Sys.time(), start_time, units="secs"))
      
      start_time <- Sys.time()
      fisher_ci <- calculate_fisher_ci(a, b, c, d, alpha)
      fisher_time <- as.numeric(difftime(Sys.time(), start_time, units="secs"))
      
      start_time <- Sys.time()
      exact_ci <- calculate_exact_ci(a, b, c, d, alpha)
      exact_time <- as.numeric(difftime(Sys.time(), start_time, units="secs"))
      
      # Store results
      result <- list(
        table = list(a=a, b=b, c=c, d=d),
        name = test_name,
        alpha = alpha,
        barnard = list(
          lower = barnard_ci$lower,
          upper = barnard_ci$upper,
          time = barnard_time
        ),
        fisher = list(
          lower = fisher_ci$lower,
          upper = fisher_ci$upper,
          time = fisher_time
        ),
        unconditional_exact = list(
          lower = exact_ci$lower,
          upper = exact_ci$upper,
          time = exact_time
        )
      )
      
      alpha_results[[length(alpha_results) + 1]] <- result
    }
    
    results[[as.character(alpha)]] <- alpha_results
  }
  
  return(results)
}

# Run tests and save results
all_results <- run_all_tests()
write_json(all_results, "r_comparison_results.json", pretty=TRUE)

# Print a summary of the results
cat("\n\n================================================================================\n")
cat("SUMMARY OF RESULTS\n")
cat("================================================================================\n\n")

for (alpha_key in names(all_results)) {
  alpha_results <- all_results[[alpha_key]]
  cat("Results for alpha =", alpha_key, "\n")
  cat("--------------------------------------------------------------------------------\n")
  
  for (result in alpha_results) {
    a <- result$table$a
    b <- result$table$b
    c <- result$table$c
    d <- result$table$d
    name <- result$name
    
    cat(sprintf("Table (%d,%d,%d,%d) - %s\n", a, b, c, d, name))
    
    barnard_ci <- result$barnard
    fisher_ci <- result$fisher
    exact_ci <- result$unconditional_exact
    
    cat(sprintf("Barnard's:       (%.6f, %.6f) in %.6f seconds\n", 
                barnard_ci$lower, barnard_ci$upper, barnard_ci$time))
    cat(sprintf("Fisher's:        (%.6f, %.6f) in %.6f seconds\n", 
                fisher_ci$lower, fisher_ci$upper, fisher_ci$time))
    cat(sprintf("Unconditional:   (%.6f, %.6f) in %.6f seconds\n\n", 
                exact_ci$lower, exact_ci$upper, exact_ci$time))
  }
  
  cat("\n")
}

cat("Results have been saved to r_comparison_results.json\n")
