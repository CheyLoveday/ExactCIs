# Script to extract and analyze the exact2x2 package implementation in R
library(exact2x2)

# Function to analyze exact2x2 implementation
analyze_exact2x2 <- function() {
  # Print method details
  cat("==== R exact2x2 Package Implementation Analysis ====\n\n")
  
  # Print available methods
  cat("Available methods in exact2x2:\n")
  print(methods("exact2x2"))
  
  # Print internals of exact2x2
  cat("\nInternal functions in exact2x2:\n")
  exact_functions <- ls("package:exact2x2")
  print(exact_functions)
  
  # Analyze specific methods
  cat("\n==== Analysis of confidence interval methods ====\n")
  
  # Function to print a function's source code
  print_function_source <- function(func_name) {
    cat(paste("\n", func_name, "source code:\n", sep=""))
    if(exists(func_name, mode="function")) {
      print(get(func_name))
    } else {
      cat("Function not directly accessible\n")
    }
  }
  
  # Try to print source code for key functions
  key_functions <- c("fisher.exact.2x2", "exact2x2", "uncond.exact", "binom.exact")
  for(func in key_functions) {
    print_function_source(func)
  }
  
  # Example table for testing
  cat("\n==== Example calculations ====\n")
  a <- 7; b <- 3; c <- 2; d <- 8
  table_mat <- matrix(c(a,c,b,d), 2, 2)
  
  cat("\nTest table:\n")
  print(table_mat)
  
  # Calculate CIs with different methods
  cat("\nFisher exact CI:\n")
  print(fisher.exact(table_mat, conf.int=TRUE))
  
  cat("\nexact2x2 method:\n")
  ex2x2_result <- exact2x2(a, b, c, d, conf.level=0.95, tsmethod="central")
  print(ex2x2_result)
  
  # Trace path of calculation for a single example
  cat("\n==== Tracing calculation steps ====\n")
  cat("For the table: [", a, b, c, d, "]\n", sep=" ")
  
  # Verbose output of exact2x2 internals (if available)
  tryCatch({
    cat("\nAttempting to trace through calculation steps...\n")
    # Use trace or debug mode if available
    options(exact2x2.debug=TRUE)
    exact2x2(a, b, c, d, conf.level=0.95, tsmethod="central")
    options(exact2x2.debug=FALSE)
  }, error=function(e) {
    cat("Debug mode not available or error in tracing:", e$message, "\n")
  })
  
  # Compare to extreme tables
  cat("\n==== Testing extreme tables ====\n")
  
  # Extreme table 1
  a1 <- 1; b1 <- 1000; c1 <- 10; d1 <- 1000
  table_mat1 <- matrix(c(a1,c1,b1,d1), 2, 2)
  
  cat("\nExtreme table 1:\n")
  print(table_mat1)
  
  cat("\nFisher exact CI:\n")
  print(fisher.exact(table_mat1, conf.int=TRUE))
  
  cat("\nexact2x2 method:\n")
  ex2x2_result1 <- exact2x2(a1, b1, c1, d1, conf.level=0.95, tsmethod="central")
  print(ex2x2_result1)
  
  # Extreme table 2
  a2 <- 10; b2 <- 1000; c2 <- 1; d2 <- 1000
  table_mat2 <- matrix(c(a2,c2,b2,d2), 2, 2)
  
  cat("\nExtreme table 2:\n")
  print(table_mat2)
  
  cat("\nFisher exact CI:\n")
  print(fisher.exact(table_mat2, conf.int=TRUE))
  
  cat("\nexact2x2 method:\n")
  ex2x2_result2 <- exact2x2(a2, b2, c2, d2, conf.level=0.95, tsmethod="central")
  print(ex2x2_result2)
  
  # Summary of algorithm
  cat("\n==== Algorithm Summary ====\n")
  cat("Based on code analysis:\n")
  cat("1. exact2x2 uses conditional exact methods (like Fisher's)\n")
  cat("2. Uses central p-values by default for confidence intervals\n")
  cat("3. Employs numerical root-finding for CI bounds\n")
  cat("4. Has special handling for sparse tables\n")
  cat("5. Different approach to handling tables with zeros\n")
}

# Run the analysis
analyze_exact2x2()
