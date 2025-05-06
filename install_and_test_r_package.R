# Script to install and demonstrate the exact2x2 R package

# Install the package if not already installed
if (!require("exact2x2")) {
  install.packages("exact2x2", repos = "https://cran.r-project.org")
}

# Load the package
library(exact2x2)

# Print package version
cat("exact2x2 package version:", packageVersion("exact2x2"), "\n")

# Test case from README: a=12, b=5, c=8, d=10
cat("\n=== Example from README: a=12, b=5, c=8, d=10 ===\n\n")
a <- 12; b <- 5; c <- 8; d <- 10

# Create the 2x2 table
table <- matrix(c(a, c, b, d), nrow=2, byrow=TRUE)
rownames(table) <- c("Group 1", "Group 2")
colnames(table) <- c("Success", "Failure")
cat("2x2 Table:\n")
print(table)

# Calculate the odds ratio
or <- (a * d) / (b * c)
cat("\nOdds ratio:", or, "\n")

# Fisher's exact test (conditional)
fisher_result <- fisher.exact(table, conf.int=TRUE, conf.level=0.95)
cat("\nFisher's exact test (conditional):\n")
cat("CI: (", fisher_result$conf.int[1], ", ", fisher_result$conf.int[2], ")\n", sep="")

# Mid-P exact test
cat("\nMid-P exact CIs:\n")

# Try different implementations of the Mid-P CI
cat("Using fisher.exact.midp function:\n")
midp_result <- fisher.exact.midp(table, conf.level=0.95)
print(midp_result)

cat("\nUsing exact2x2 function with midp=TRUE:\n")
midp_result2 <- exact2x2(table, midp=TRUE, conf.level=0.95)
print(midp_result2)

# Show available methods
cat("\nAvailable methods in exact2x2 package:\n")
print(methods(class="exact2x2"))

# Show help for the fisher.exact.midp function
cat("\nHelp for fisher.exact.midp function:\n")
help(fisher.exact.midp)
