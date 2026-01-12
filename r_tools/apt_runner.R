#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
in_path <- args[1]
out_path <- args[2]

suppressWarnings(suppressMessages({
  library(jsonlite)
  library(apt)   # CRAN package "apt" (lowercase)
}))

write_out <- function(obj) {
  writeLines(jsonlite::toJSON(obj, auto_unbox = TRUE, na = "null", digits = 10),
             out_path, useBytes = TRUE)
}

as_num_ts <- function(v, freq = 1) {
  # robust conversion to numeric vector -> ts
  vv <- as.numeric(v)
  vv <- vv[is.finite(vv)]
  stats::ts(vv, frequency = freq)
}

safe_capture <- function(expr) {
  paste(capture.output(expr), collapse = "\n")
}

tryCatch({
  inp <- jsonlite::fromJSON(in_path)

  mode <- inp$mode
  model <- inp$model
  lag <- as.integer(inp$lag)
  thresh <- as.numeric(inp$thresh)
  maxlag <- as.integer(inp$maxlag)
  adjust <- isTRUE(inp$adjust)
  th_range <- inp$th_range
  split <- isTRUE(inp$split)
  which_fit <- inp$which
  freq <- inp$frequency
  if (is.null(freq) || !is.finite(freq)) freq <- 1

  y_raw <- inp$data$y
  x_raw <- inp$data$x

  y <- as_num_ts(y_raw, freq = freq)
  x <- as_num_ts(x_raw, freq = freq)

  if (length(y) < 10 || length(x) < 10) stop("Too few observations after cleaning.")
  if (length(y) != length(x)) {
    n <- min(length(y), length(x))
    y <- y[seq_len(n)]
    x <- x[seq_len(n)]
  }

  # hard check: apt wants ts
  if (!stats::is.ts(y) || !stats::is.ts(x)) stop("y/x are not ts after conversion (unexpected).")

  out <- list(
    ok = TRUE,
    mode = mode,
    apt_version = as.character(utils::packageVersion("apt")),
    debug = list(class_y = class(y), class_x = class(x), n = length(y), frequency = freq)
  )

  fit <- NULL

  # ---- dispatch ----
  if (mode == "ciTarFit") {
    # ciTarFit(y, x, model, lag, thresh, small.win)
    small_win <- inp$small_win
    if (is.null(small_win) || !is.finite(as.numeric(small_win))) {
      fit <- apt::ciTarFit(y = y, x = x, model = model, lag = lag, thresh = thresh)
    } else {
      fit <- apt::ciTarFit(y = y, x = x, model = model, lag = lag, thresh = thresh, small.win = as.numeric(small_win))
    }

  } else if (mode == "ciTarLag") {
    # ciTarLag(y, x, model, maxlag, adjust)
    fit <- apt::ciTarLag(y = y, x = x, model = model, maxlag = maxlag, adjust = adjust)

  } else if (mode == "ciTarThd") {
    # ciTarThd(y, x, model, lag, thRange)
    if (is.null(th_range) || length(th_range) != 2) stop("th_range must be length-2 (min,max).")
    thr <- as.numeric(th_range)
    fit <- apt::ciTarThd(y = y, x = x, model = model, lag = lag, thRange = thr)

  } else if (mode == "ecmSymFit") {
    # ecmSymFit(y, x, lag)  (NO digits in apt 4.0)
    fit <- apt::ecmSymFit(y = y, x = x, lag = lag)

  } else if (mode == "ecmAsyFit") {
    # ecmAsyFit(y, x, lag, split)
    fit <- apt::ecmAsyFit(y = y, x = x, lag = lag, split = split)

  } else if (mode == "ecmAsyTest") {
    fit0 <- apt::ecmAsyFit(y = y, x = x, lag = lag, split = split)
    tst <- apt::ecmAsyTest(fit0)
    out$test <- safe_capture(print(tst))
    fit <- fit0

  } else if (mode == "ecmDiag") {
    if (!is.null(which_fit) && which_fit == "asy") {
      fit0 <- apt::ecmAsyFit(y = y, x = x, lag = lag, split = split)
      dg <- apt::ecmDiag(fit0)
      out$diag <- safe_capture(print(dg))
      fit <- fit0
    } else {
      fit0 <- apt::ecmSymFit(y = y, x = x, lag = lag)
      dg <- apt::ecmDiag(fit0)
      out$diag <- safe_capture(print(dg))
      fit <- fit0
    }
  } else {
    stop(paste0("Unknown mode: ", mode))
  }

  if (!is.null(fit)) {
    out$summary <- safe_capture(print(fit))
  }

  write_out(out)

}, error = function(e) {
  write_out(list(ok = FALSE, error = as.character(e)))
})
