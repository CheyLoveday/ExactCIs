# R script to compute exact confidence intervals for 2x2 tables
# using the exact2x2 package for reference comparison

# Install exact2x2 package if needed (uncomment if required)
# install.packages("exact2x2")

library(exact2x2)

# Function to compute and print CIs for a 2x2 table
compute_all_cis <- function(a, b, c, d, alpha=0.05) {
  # Create the table
  table <- matrix(c(a, c, b, d), nrow=2)
  
  cat("Table:\n")
  print(table)
  cat("\n")
  
  # Compute odds ratio
  or <- (a * d) / (b * c)
  cat(sprintf("Odds ratio: %.4f\n\n", or))
  
  # Fisher's exact test (conditional)
  fisher <- fisher.exact(table, conf.int=TRUE, conf.level=1-alpha)
  cat("Fisher's exact (conditional):\n")
  cat(sprintf("CI: (%.4f, %.4f)\n\n", fisher$conf.int[1], fisher$conf.int[2]))
  
  # Mid-P adjusted CI
  # Use exact2x2 function with midp=TRUE
  tryCatch({
    midp <- exact2x2(a, b, c, d, midp=TRUE, conf.level=1-alpha)
    cat("Mid-P adjusted:\n")
    cat(sprintf("CI: (%.4f, %.4f)\n\n", midp$conf.int[1], midp$conf.int[2]))
  }, error = function(e) {
    cat("Mid-P adjusted:\n")
    cat("Error computing Mid-P CI:", e$message, "\n\n")
  })
  
  # Try alternative midp methods
  tryCatch({
    midp_alt <- fisher.exact.midp(a, b, c, d, conf.level=1-alpha)
    cat("Mid-P adjusted (alternative):\n")
    cat(sprintf("CI: (%.4f, %.4f)\n\n", midp_alt$conf.int[1], midp_alt$conf.int[2]))
  }, error = function(e) {
    cat("Mid-P adjusted (alternative):\n")
    cat("Error computing Mid-P CI:", e$message, "\n\n")
  })
  
  # Blaker's exact CI if available
  tryCatch({
    blaker <- exact2x2(a, b, c, d, method="blaker", conf.level=1-alpha)
    cat("Blaker's exact:\n")
    cat(sprintf("CI: (%.4f, %.4f)\n\n", blaker$conf.int[1], blaker$conf.int[2]))
  }, error = function(e) {
    cat("Blaker's method:\n")
    cat("Error computing Blaker CI:", e$message, "\n\n")
  })
  
  # Unconditional exact (if available)
  tryCatch({
    uncond <- exact2x2(a, b, c, d, method="uncond", conf.level=1-alpha)
    cat("Unconditional exact:\n")
    cat(sprintf("CI: (%.4f, %.4f)\n\n", uncond$conf.int[1], uncond$conf.int[2]))
  }, error = function(e) {
    cat("Unconditional method:\n")
    cat("Error computing Unconditional CI:", e$message, "\n\n")
  })
  
  # Wald CI with Haldane-Anscombe correction
  tryCatch({
    # Use the confidence interval method from the epitools package if available
    if (requireNamespace("epitools", quietly = TRUE)) {
      wald <- epitools::oddsratio.midp(table, conf.level=1-alpha)
      cat("Wald with Haldane-Anscombe correction:\n")
      cat(sprintf("CI: (%.4f, %.4f)\n\n", wald$OR[1,2], wald$OR[1,3]))
    } else {
      # Fallback to a simple calculation
      a_adj <- a + 0.5
      b_adj <- b + 0.5
      c_adj <- c + 0.5
      d_adj <- d + 0.5
      or_adj <- (a_adj * d_adj) / (b_adj * c_adj)
      se <- sqrt(1/a_adj + 1/b_adj + 1/c_adj + 1/d_adj)
      z <- qnorm(1 - alpha/2)
      lo <- exp(log(or_adj) - z*se)
      hi <- exp(log(or_adj) + z*se)
      cat("Wald with Haldane-Anscombe correction (calculated):\n")
      cat(sprintf("CI: (%.4f, %.4f)\n\n", lo, hi))
    }
  }, error = function(e) {
    cat("Wald with Haldane-Anscombe correction:\n")
    cat("Error computing Wald CI:", e$message, "\n\n")
  })
}

# Test cases
cat("\n=== Example from README: a=12, b=5, c=8, d=10 ===\n\n")
compute_all_cis(12, 5, 8, 10)

cat("\n=== Small counts: a=1, b=1, c=1, d=1 ===\n\n")
compute_all_cis(1, 1, 1, 1)

cat("\n=== Zero in one cell: a=0, b=5, c=8, d=10 ===\n\n")
compute_all_cis(0, 5, 8, 10)

cat("\n=== Large imbalance: a=50, b=5, c=2, d=20 ===\n\n")
compute_all_cis(50, 5, 2, 20)
