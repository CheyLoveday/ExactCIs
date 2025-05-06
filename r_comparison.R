#!/usr/bin/env Rscript

# Load required libraries
if (!requireNamespace("exact2x2", quietly = TRUE)) {
  install.packages("exact2x2", repos = "http://cran.us.r-project.org")
}
if (!requireNamespace("jsonlite", quietly = TRUE)) {
  install.packages("jsonlite", repos = "http://cran.us.r-project.org")
}
library(exact2x2)
library(jsonlite)

# Test cases matching our Python tests
test_cases <- list(
  list(a=3, b=1, c=1, d=3, name="Std symmetric (OR=9)"),
  list(a=1, b=9, c=9, d=1, name="Std symmetric (OR approx 0.0123)"),
  list(a=5, b=0, c=2, d=8, name="Zero cell b=0"),
  list(a=0, b=5, c=8, d=2, name="Zero cell a=0"),
  list(a=10, b=7, c=9990, d=9993, name="Large N, OR approx 1.429"),
  list(a=2, b=8, c=8, d=2, name="Symmetric, OR=0.0625 (R exact2x2 ex.)"),
  list(a=1, b=1, c=1, d=20, name="Asymmetric, OR=20")
)

# Alpha value to test
alpha_to_test <- 0.05

# Function to calculate Blaker CIs using exact2x2
calculate_blaker_ci <- function(a, b, c, d, alpha = 0.05) {
  tryCatch({
    m <- matrix(c(a, c, b, d), nrow=2, byrow=TRUE) # exact2x2 expects a,c,b,d for rows
    result <- exact2x2::exact2x2(m, 
                                  tsmethod="blaker", 
                                  conf.level=1-alpha,
                                  alternative="two.sided",
                                  plot=FALSE) # Ensure plot is FALSE for non-interactive
    
    ci <- result$conf.int
    estimate <- result$estimate # Odds Ratio estimate
    return(list(lower=ci[1], upper=ci[2], estimate=estimate))
  }, error = function(e) {
    cat("Error in Blaker's CI for table (", a, b, c, d, "): ", e$message, "\n")
    return(list(lower=NA, upper=NA, estimate=NA))
  })
}

# Function to run all tests
run_blaker_comparison_tests <- function() {
  results_list <- list()
  
  alpha <- alpha_to_test # Use the single alpha value
  
  for (test_case in test_cases) {
    a <- test_case$a
    b <- test_case$b
    c <- test_case$c
    d <- test_case$d
    name <- test_case$name
    
    test_name <- paste0(name, " (alpha=", alpha, ")")
    
    cat("Testing R Blaker CI for:", test_name, " Table: (a=",a,", b=",b,", c=",c,", d=",d,")...\n")
    
    # Get Blaker results
    start_time <- Sys.time()
    blaker_result <- calculate_blaker_ci(a, b, c, d, alpha)
    blaker_time <- as.numeric(difftime(Sys.time(), start_time, units="secs"))
    
    # Store results
    current_result <- list(
      table_desc = name,
      table_values = list(a=a, b=b, c=c, d=d),
      alpha = alpha,
      blaker_ci_r = list(
        lower = blaker_result$lower,
        upper = blaker_result$upper,
        estimate = blaker_result$estimate,
        time_r = blaker_time
      )
    )
    
    results_list[[length(results_list) + 1]] <- current_result
  }
  
  return(results_list)
}

# Run tests and save results
blaker_comparison_results <- run_blaker_comparison_tests()
output_filename <- "r_blaker_comparison_results.json"
write_json(blaker_comparison_results, output_filename, pretty=TRUE, auto_unbox=TRUE)

# Print a summary of the results
cat("\n\n================================================================================\n")
cat("SUMMARY OF R BLAKER CI RESULTS\n")
cat("================================================================================\n\n")

for (result in blaker_comparison_results) {
  a <- result$table_values$a
  b <- result$table_values$b
  c <- result$table_values$c
  d <- result$table_values$d
  name <- result$table_desc
  alpha_val <- result$alpha
  
  cat(sprintf("Table: %s (a=%d, b=%d, c=%d, d=%d), Alpha=%.2f\n", name, a, b, c, d, alpha_val))
  
  r_blaker_ci <- result$blaker_ci_r
  
  cat(sprintf("R Blaker CI: Lower=%.6f, Upper=%.6f, Estimate=%.6f (Time: %.6f s)\n\n", 
              r_blaker_ci$lower, r_blaker_ci$upper, r_blaker_ci$estimate, r_blaker_ci$time_r))
}

cat("R Blaker CI comparison results have been saved to", output_filename, "\n")
