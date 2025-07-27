# R script to calculate Fisher's exact test CIs for specific tables
library(jsonlite)

calculate_fisher_ci <- function(a, b, c, d, conf_level = 0.95) {
  mat <- matrix(c(a, b, c, d), nrow = 2, byrow = TRUE)
  test_result <- fisher.test(mat, conf.level = conf_level)
  ci <- test_result$conf.int
  # R can return 'Inf' as a string for infinity in some contexts, or it's a numeric Inf.
  # Ensure we handle it consistently for JSON output (e.g. as string "Inf" or R's Inf will be null if not handled in toJSON)
  # jsonlite's toJSON typically handles Inf correctly by converting to null by default, or a string if specified by I()
  # Let's ensure they are numeric and then rounded, Inf will remain Inf.
  lower_val <- ci[1]
  upper_val <- ci[2]
  
  return(list(lower = round(lower_val, 6), 
                upper = if(is.infinite(upper_val)) "Inf" else round(upper_val, 6)))
}

tables <- list(
  list(name = "Table 1 (5,2,9995,9998)", data = list(a=5, b=2, c=9995, d=9998)),
  list(name = "Table 2 (10,7,9990,9993)", data = list(a=10, b=7, c=9990, d=9993)),
  list(name = "Table 3 (3,0,9997,10000)", data = list(a=3, b=0, c=9997, d=10000))
)

results <- list()

for (i in 1:length(tables)) {
  table_item <- tables[[i]]
  data <- table_item$data
  ci_res <- calculate_fisher_ci(data$a, data$b, data$c, data$d)
  results[[table_item$name]] <- ci_res
}

# Output results as JSON
json_output <- toJSON(results, pretty = TRUE, auto_unbox = TRUE)
cat(json_output)
