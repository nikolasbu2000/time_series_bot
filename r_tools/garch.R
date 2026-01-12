#!/usr/bin/env Rscript

suppressWarnings(suppressMessages({
  library(jsonlite)
}))

args <- commandArgs(trailingOnly = TRUE)

# Expect: Rscript garch.R meta.json data.csv out.json
if (length(args) < 3) {
  cat(toJSON(list(ok=FALSE, error="Usage: Rscript garch.R <meta.json> <data.csv> <out.json>"), auto_unbox=TRUE))
  quit(status=2)
}

meta_path <- args[[1]]
data_path <- args[[2]]
out_path  <- args[[3]]

write_out <- function(obj) {
  # Always write UTF-8 JSON to out_path
  writeLines(jsonlite::toJSON(obj, auto_unbox=TRUE, na="null", digits=10), out_path, useBytes=TRUE)
}

meta <- tryCatch(jsonlite::fromJSON(meta_path), error=function(e) NULL)
df   <- tryCatch(read.csv(data_path), error=function(e) NULL)

if (is.null(meta) || is.null(df)) {
  write_out(list(ok=FALSE, error="Could not read meta or data"))
  quit(status=2)
}

mode <- meta$mode
if (is.null(mode)) mode <- "ugarch"

out <- tryCatch({

  if (mode == "ugarch") {
    suppressMessages(library(rugarch))

    if (!("y" %in% names(df))) stop("data.csv must have column y")
    y <- as.numeric(df$y)

    variance_model <- meta$variance_model; if (is.null(variance_model)) variance_model <- "sGARCH"
    garch_order_p  <- meta$garch_p;        if (is.null(garch_order_p))  garch_order_p  <- 1
    garch_order_q  <- meta$garch_q;        if (is.null(garch_order_q))  garch_order_q  <- 1

    mean_arma_p <- meta$arma_p; if (is.null(mean_arma_p)) mean_arma_p <- 0
    mean_arma_q <- meta$arma_q; if (is.null(mean_arma_q)) mean_arma_q <- 0
    include_mean <- meta$include_mean; if (is.null(include_mean)) include_mean <- TRUE
    dist_model <- meta$dist; if (is.null(dist_model)) dist_model <- "norm"

    spec <- ugarchspec(
      variance.model = list(model=variance_model, garchOrder=c(garch_order_p, garch_order_q)),
      mean.model = list(armaOrder=c(mean_arma_p, mean_arma_q), include.mean=include_mean),
      distribution.model = dist_model
    )

    fit <- ugarchfit(spec=spec, data=y, solver="hybrid")
    s <- capture.output(show(fit))

    list(
      ok=TRUE,
      mode=mode,
      variance_model=variance_model,
      garch_order=c(garch_order_p, garch_order_q),
      arma_order=c(mean_arma_p, mean_arma_q),
      dist=dist_model,
      coef=as.list(coef(fit)),
      ic=as.list(infocriteria(fit)),
      summary_text=paste(s, collapse="\n")
    )

  } else if (mode == "dcc") {
    suppressMessages(library(rugarch))
    suppressMessages(library(rmgarch))

    cols <- meta$cols
    if (is.null(cols) || length(cols) < 2) cols <- names(df)
    if (length(cols) < 2) stop("Need at least 2 series for DCC MGARCH")

    X <- as.matrix(df[, cols, drop=FALSE])
    X <- apply(X, 2, as.numeric)

    variance_model <- meta$variance_model; if (is.null(variance_model)) variance_model <- "sGARCH"
    garch_order_p  <- meta$garch_p;        if (is.null(garch_order_p))  garch_order_p  <- 1
    garch_order_q  <- meta$garch_q;        if (is.null(garch_order_q))  garch_order_q  <- 1
    dist_model <- meta$dist; if (is.null(dist_model)) dist_model <- "norm"

    uspec <- ugarchspec(
      variance.model=list(model=variance_model, garchOrder=c(garch_order_p, garch_order_q)),
      mean.model=list(armaOrder=c(0,0), include.mean=TRUE),
      distribution.model=dist_model
    )

    mspec <- multispec(replicate(ncol(X), uspec))
    dcc_order <- meta$dcc_order; if (is.null(dcc_order)) dcc_order <- c(1,1)

    spec <- dccspec(uspec=mspec, dccOrder=dcc_order, distribution="mvnorm")
    fit <- dccfit(spec, data=X, fit.control=list(eval.se=TRUE))

    s <- capture.output(show(fit))

    list(
      ok=TRUE,
      mode=mode,
      cols=cols,
      dcc_order=as.list(dcc_order),
      summary_text=paste(s, collapse="\n")
    )

  } else {
    list(ok=FALSE, error=paste0("Unknown mode: ", mode))
  }

}, error=function(e) {
  list(ok=FALSE, error=as.character(e))
})

write_out(out)
