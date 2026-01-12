#!/usr/bin/env Rscript

suppressWarnings(suppressMessages({
  library(jsonlite)
}))

ok_json <- function(x) {
  cat(toJSON(x, auto_unbox = TRUE, null = "null"))
}

fail_json <- function(msg, warnings = character(), messages = character(), debug = list()) {
  ok_json(list(ok = FALSE, error = msg, warnings = warnings, messages = paste(messages, collapse = "\n"), debug = debug))
  quit(status = 0)
}

args <- commandArgs(trailingOnly = TRUE)

# Expected args:
# 1) data_csv
# 2) meta_json  (contains formula, ic, maxlag, case, graph)
if (length(args) < 2) {
  fail_json("Usage: nardl.R <data.csv> <meta.json>")
}

data_path <- args[1]
meta_path <- args[2]

# capture warnings/messages
warns <- character()
msgs  <- character()
w_handler <- function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") }
m_handler <- function(m) { msgs  <<- c(msgs,  conditionMessage(m)); invokeRestart("muffleMessage") }

withCallingHandlers({

  meta <- fromJSON(meta_path)

  if (is.null(meta$formula) || is.na(meta$formula) || meta$formula == "") {
    fail_json("Formula is NA/empty. Check meta$formula.", debug = list(meta_formula = meta$formula))
  }

  ic     <- if (!is.null(meta$ic))     tolower(meta$ic) else "aic"
  maxlag <- if (!is.null(meta$maxlag)) as.integer(meta$maxlag) else 4L
  case   <- if (!is.null(meta$case))   as.integer(meta$case) else 3L
  graph  <- if (!is.null(meta$graph))  as.logical(meta$graph) else FALSE

  d <- read.csv(data_path, check.names = FALSE)

  # ---- Try to load a NARDL implementation ----
  # We try (in this order):
  # 1) package "nardl" providing nardl()
  # 2) package "ARDL" providing nardl()
  have_nardl_pkg <- requireNamespace("nardl", quietly = TRUE)
  have_ardl_pkg  <- requireNamespace("ARDL", quietly = TRUE)

  fit <- NULL
  fit_class <- NULL

  if (have_nardl_pkg && exists("nardl", where = asNamespace("nardl"), mode = "function")) {
    f <- get("nardl", envir = asNamespace("nardl"))
    fit <- f(as.formula(meta$formula), data = d, ic = ic, maxlag = maxlag, case = case, graph = graph)
    fit_class <- paste(class(fit), collapse = ",")
  } else if (have_ardl_pkg && exists("nardl", where = asNamespace("ARDL"), mode = "function")) {
    f <- get("nardl", envir = asNamespace("ARDL"))
    fit <- f(as.formula(meta$formula), data = d, ic = ic, maxlag = maxlag, case = case, graph = graph)
    fit_class <- paste(class(fit), collapse = ",")
  } else {
    fail_json("No NARDL function found. Install an R package that provides nardl(): try install.packages('nardl') or install.packages('ARDL').")
  }

  # ---- Output: robust summary capture ----
  sum_txt <- tryCatch(
    paste(capture.output(print(summary(fit))), collapse = "\n"),
    error = function(e) paste("Could not capture summary:", conditionMessage(e))
  )

  # Some packages store useful components; we expose whatever exists safely
  extract_safe <- function(obj, name) {
    if (!is.null(obj[[name]])) return(obj[[name]])
    return(NULL)
  }

  out <- list(
    ok = TRUE,
    engine = "R",
    function_used = "nardl()",
    fit_class = fit_class,
    formula = meta$formula,
    ic = ic,
    maxlag = maxlag,
    case = case,
    summary = sum_txt,
    components = list(
      long_run = extract_safe(fit, "long_run"),
      short_run = extract_safe(fit, "short_run"),
      bounds = extract_safe(fit, "bounds"),
      diagnostics = extract_safe(fit, "diagnostics")
    )
  )

  ok_json(out)

}, warning = w_handler, message = m_handler)

# If messages/warnings exist, they are already captured; JSON printed once above.
