# R script to calculate CIs for specific tables using various methods

# Load necessary libraries
# Ensure 'exactci' and 'exact2x2' are installed: install.packages(c("exactci", "exact2x2"))
library(exactci) # exact2x2 depends on this
library(exact2x2)
library(jsonlite)

# Function to install a package if it's not already installed
ensure_package <- function(pkg_name) {
  if (!requireNamespace(pkg_name, quietly = TRUE)) {
    message(paste("Installing package:", pkg_name))
    install.packages(pkg_name, repos = "http://cran.us.r-project.org")
  }
  library(pkg_name, character.only = TRUE)
}

# Ensure jsonlite and exact2x2 are available
ensure_package("jsonlite")
ensure_package("exact2x2") # For Blaker and Mid-P
ensure_package("exactci")

# Helper to create matrix
create_matrix <- function(a,b,c,d) {
  matrix(c(a, b, c, d), nrow = 2, byrow = TRUE)
}

# --- CI Calculation Functions ---

# Fisher Exact (from base R)
calculate_r_fisher_ci <- function(a, b, c, d, conf_level = 0.95) {
  mat <- create_matrix(a,b,c,d)
  tryCatch({
    test_result <- fisher.test(mat, conf.level = conf_level)
    ci <- test_result$conf.int
    list(lower = round(ci[1], 6), upper = if(is.infinite(ci[2])) "Inf" else round(ci[2], 6))
  }, error = function(e) list(lower = as.character(e), upper = as.character(e)))
}

# Wald Logit with Haldane-Anscombe correction
calculate_r_wald_logit_ci <- function(a, b, c, d, conf_level = 0.95) {
  # Add 0.5 to each cell (Haldane-Anscombe correction)
  a_adj <- a + 0.5
  b_adj <- b + 0.5
  c_adj <- c + 0.5
  d_adj <- d + 0.5
  
  or <- (a_adj * d_adj) / (b_adj * c_adj)
  log_or <- log(or)
  se_log_or <- sqrt(1/a_adj + 1/b_adj + 1/c_adj + 1/d_adj)
  
  z <- qnorm(1 - (1 - conf_level) / 2)
  log_lower <- log_or - z * se_log_or
  log_upper <- log_or + z * se_log_or
  
  lower_ci <- exp(log_lower)
  upper_ci <- exp(log_upper)
  
  return(list(lower = round(lower_ci, 6), upper = round(upper_ci, 6)))
}

# Function to calculate Blaker CI using exact2x2 package
calculate_r_blaker_ci <- function(a, b, c, d, alpha = 0.05) {
  tryCatch({
    mat <- create_matrix(a,b,c,d) # blaker.exact expects a matrix
    # Using blaker.exact from exact2x2 package
    ci_obj <- exact2x2::blaker.exact(mat, conf.level = 1 - alpha)
    ci <- ci_obj$conf.int
    return(list(lower = round(ci[1], 6), upper = round(ci[2], 6)))
  }, error = function(e) {
    return(list(lower = e$message, upper = e$message))
  })
}

# Function to calculate Mid-P CI using exact2x2 package
calculate_r_midp_conditional_ci <- function(a, b, c, d, alpha = 0.05) {
  tryCatch({
    mat <- create_matrix(a,b,c,d)
    # Using midp = TRUE and tsmethod = "central" in exact2x2 function from the exact2x2 package
    ci_obj <- exact2x2::exact2x2(mat, midp = TRUE, tsmethod = "central", conf.level = 1 - alpha)
    ci <- ci_obj$conf.int
    return(list(lower = round(ci[1], 6), upper = round(ci[2], 6)))
  }, error = function(e) {
    return(list(lower = e$message, upper = e$message))
  })
}

# --- Main Execution ---
tables <- list(
  list(name = "Table 1 (5,2,9995,9998)", data = list(a=5, b=2, c=9995, d=9998)),
  list(name = "Table 2 (10,7,9990,9993)", data = list(a=10, b=7, c=9990, d=9993)),
  list(name = "Table 3 (3,0,9997,10000)", data = list(a=3, b=0, c=9997, d=10000))
)

all_r_results <- list()

for (i in 1:length(tables)) {
  table_item <- tables[[i]]
  data <- table_item$data
  table_name <- table_item$name
  
  all_r_results[[table_name]] <- list(
    R_Fisher_Exact = calculate_r_fisher_ci(data$a, data$b, data$c, data$d),
    R_Wald_Haldane = calculate_r_wald_logit_ci(data$a, data$b, data$c, data$d), 
    R_Blaker_Exact = calculate_r_blaker_ci(data$a, data$b, data$c, data$d),
    R_MidP_Conditional = calculate_r_midp_conditional_ci(data$a, data$b, data$c, data$d)
  )
}

# Output results as JSON
json_output <- toJSON(all_r_results, pretty = TRUE, auto_unbox = TRUE)
cat(json_output)
