#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
data_path <- args[1]
meta_path <- args[2]

suppressMessages({
  library(jsonlite)
  library(ARDL)
})

emit <- function(x) {
  cat("__JSON__")
  cat(toJSON(x, auto_unbox = TRUE, null = "null", digits = 12))
}

fail <- function(msg, debug = list()) {
  emit(list(ok = FALSE, error = msg, debug = debug))
  quit(status = 0)
}

if (is.na(data_path) || data_path == "" || !file.exists(data_path)) {
  fail("data.csv not found", list(data_path = data_path))
}
if (is.na(meta_path) || meta_path == "" || !file.exists(meta_path)) {
  fail("meta.json not found", list(meta_path = meta_path))
}

meta <- tryCatch(fromJSON(meta_path), error = function(e) NULL)
if (is.null(meta)) {
  fail("Could not read meta.json")
}

if (is.null(meta$formula) || is.na(meta$formula) || meta$formula == "") {
  fail("Formula is NA/empty. Check Python meta$formula and selected variables.", list(meta_formula = meta$formula))
}

df <- tryCatch(read.csv(data_path), error = function(e) NULL)
if (is.null(df)) {
  fail("Could not read data.csv")
}

ic <- if (!is.null(meta$ic)) tolower(meta$ic) else "aic"
alpha <- if (!is.null(meta$alpha)) as.numeric(meta$alpha) else 0.05
exact <- if (!is.null(meta$exact)) as.logical(meta$exact) else FALSE
Rreps <- if (!is.null(meta$R)) as.integer(meta$R) else 2000

max_order <- meta$max_order
# Keep it flexible: allow scalar or vector. auto_ardl accepts numeric vector too.
# If scalar is given, it applies to all vars internally.
# We'll pass through as-is.

bounds_cases <- meta$bounds_cases
if (is.null(bounds_cases) || length(bounds_cases) == 0) {
  bounds_cases <- c("auto")
}

# Fit auto ARDL
fit_auto <- tryCatch(
  auto_ardl(
    formula = as.formula(meta$formula),
    data = df,
    max_order = max_order,
    selection = ic
  ),
  error = function(e) e
)

if (inherits(fit_auto, "error")) {
  fail("auto_ardl failed", list(message = conditionMessage(fit_auto)))
}

best <- fit_auto$best_model
if (is.null(best)) {
  fail("auto_ardl returned no best_model (unexpected).")
}

# Basic outputs
sum_txt <- paste(capture.output(summary(best)), collapse = "\n")
order_txt <- tryCatch(paste(fit_auto$best_order, collapse = ","), error = function(e) "")

# Bounds tests (F and t) for selected cases.
# We allow: "auto" and numeric 1..5.
run_bounds_for_case <- function(case_val) {
  # bounds_f_test expects a case that is compatible with the model (trend/intercept)
  # We catch errors and report them.
  f_res <- tryCatch(
    bounds_f_test(best, case = case_val, alpha = alpha, exact = exact, R = Rreps),
    error = function(e) e
  )
  t_res <- tryCatch(
    bounds_t_test(best, case = case_val, alpha = alpha, exact = exact, R = Rreps),
    error = function(e) e
  )

  out <- list(case = case_val)

  if (inherits(f_res, "error")) {
    out$f_ok <- FALSE
    out$f_error <- conditionMessage(f_res)
  } else {
    out$f_ok <- TRUE
    out$f_stat <- unname(f_res$statistic)
    out$f_pvalue <- if (!is.null(f_res$p.value)) unname(f_res$p.value) else NA
    out$f_tab <- tryCatch(as.data.frame(f_res$tab), error = function(e) NULL)
  }

  if (inherits(t_res, "error")) {
    out$t_ok <- FALSE
    out$t_error <- conditionMessage(t_res)
  } else {
    out$t_ok <- TRUE
    out$t_stat <- unname(t_res$statistic)
    out$t_pvalue <- if (!is.null(t_res$p.value)) unname(t_res$p.value) else NA
    out$t_tab <- tryCatch(as.data.frame(t_res$tab), error = function(e) NULL)
  }

  out
}

bounds_out <- list()
for (cc in bounds_cases) {
  case_val <- cc
  if (!is.character(case_val)) case_val <- as.character(case_val)

  if (tolower(case_val) == "auto") {
    # If case="auto" isn't accepted by ARDL package, we instead try sensible defaults 1..5 and keep those that work.
    # But ARDL::bounds_* requires explicit case; "auto" is not documented there.
    # We'll interpret "auto" as: try 1..5 and return successes/failures.
    for (k in 1:5) {
      bounds_out[[paste0("case_", k)]] <- run_bounds_for_case(k)
    }
  } else {
    # numeric case
    k <- suppressWarnings(as.integer(case_val))
    if (is.na(k)) {
      bounds_out[[paste0("case_", case_val)]] <- list(case = case_val, f_ok = FALSE, t_ok = FALSE,
                                                     f_error = "Invalid case (not int 1..5)",
                                                     t_error = "Invalid case (not int 1..5)")
    } else {
      bounds_out[[paste0("case_", k)]] <- run_bounds_for_case(k)
    }
  }
}

# Emit JSON
emit(list(
  ok = TRUE,
  meta = list(
    formula = meta$formula,
    ic = ic,
    max_order = max_order,
    n_obs = nrow(df),
    best_order = order_txt,
    bounds_cases = bounds_cases,
    alpha = alpha,
    exact = exact,
    R = Rreps
  ),
  summary = sum_txt,
  bounds = bounds_out
))
