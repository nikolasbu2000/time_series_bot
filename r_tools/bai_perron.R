args <- commandArgs(trailingOnly = TRUE)

# args:
# 1 y_csv, 2 model, 3 max_breaks, 4 ic, 5 eps1, 6 mode, 7 k_fixed
y_csv      <- args[1]
model      <- args[2]                  # "level" or "trend"
max_breaks <- as.integer(args[3])      # max m
ic         <- args[4]                  # "KT" | "BIC" | "LWZ"
eps1       <- as.numeric(args[5])      # trimming, e.g. 0.15
mode       <- args[6]                  # "unknown" | "fixed"
k_fixed    <- as.integer(args[7])      # only used if mode="fixed"

suppressPackageStartupMessages(library(mbreaks))
suppressPackageStartupMessages(library(jsonlite))

# Read series
d <- read.csv(y_csv)
y <- d[[1]]
y <- y[!is.na(y)]
n <- length(y)

if (n < 20) {
  cat(toJSON(list(error="Series too short (<20)"), auto_unbox=TRUE))
  quit(status=0)
}

# Data for trend option
df <- data.frame(y=y, t=1:n)

# Configure level vs trend
# - level: breaks in intercept (mean shifts)
# - trend: breaks in intercept + slope on t (trend shifts)
z_name <- NULL
const  <- 1
if (model == "trend") {
  z_name <- c("t")
  const  <- 1
} else if (model == "level") {
  z_name <- NULL
  const  <- 1
} else {
  cat(toJSON(list(error="model must be 'level' or 'trend'"), auto_unbox=TRUE))
  quit(status=0)
}

pick_by_ic <- function(res_all, ic) {
  if (ic == "KT")  return(res_all$KT)
  if (ic == "BIC") return(res_all$BIC)
  if (ic == "LWZ") return(res_all$LWZ)
  stop("ic must be KT/BIC/LWZ")
}

# Run full procedure (unknown breaks up to m)
res_all <- mdl(
  y_name   = "y",
  z_name   = z_name,
  x_name   = NULL,
  data     = df,
  m        = max_breaks,
  eps1     = eps1,
  robust   = 1,
  prewhit  = 1,
  const    = const
)

sel_obj <- pick_by_ic(res_all, ic)

# --- helper: try to extract model for fixed k ---
extract_fixed_model <- function(sel_obj, k_fixed) {
  # Common patterns across versions:
  # 1) sel_obj$models is a list indexed by k+1
  if (!is.null(sel_obj$models) && is.list(sel_obj$models)) {
    idx <- k_fixed + 1
    if (idx >= 1 && idx <= length(sel_obj$models)) return(sel_obj$models[[idx]])
  }
  # 2) sel_obj is itself a list of candidate models over k
  if (is.list(sel_obj) && length(sel_obj) >= (k_fixed + 1)) {
    # Heuristic: if elements look like model objects
    idx <- k_fixed + 1
    cand <- sel_obj[[idx]]
    if (!is.null(cand)) return(cand)
  }
  # 3) Sometimes nested under $out or similar
  if (!is.null(sel_obj$out) && is.list(sel_obj$out) && length(sel_obj$out) >= (k_fixed + 1)) {
    idx <- k_fixed + 1
    cand <- sel_obj$out[[idx]]
    if (!is.null(cand)) return(cand)
  }
  return(NULL)
}

selected_raw <- NULL

if (mode == "unknown") {
  # sel_obj is already the selected model (best by IC)
  selected_raw <- sel_obj
} else if (mode == "fixed") {
  if (is.na(k_fixed) || k_fixed < 0) {
    cat(toJSON(list(error="k_fixed must be >= 0 for mode='fixed'"), auto_unbox=TRUE))
    quit(status=0)
  }
  if (k_fixed > max_breaks) {
    cat(toJSON(list(error="k_fixed cannot be > max_breaks"), auto_unbox=TRUE))
    quit(status=0)
  }
  selected_raw <- extract_fixed_model(sel_obj, k_fixed)
  if (is.null(selected_raw)) {
    # Provide debug info to help adjust selector for your mbreaks version
    dbg <- paste(capture.output(str(sel_obj, max.level=2)), collapse="\n")
    cat(toJSON(list(
      error="Cannot extract fixed-k model from mbreaks object (version-dependent structure).",
      hint="Run with mode='unknown' or share 'sel_obj' structure; selector can be adapted.",
      sel_obj_structure=dbg
    ), auto_unbox=TRUE))
    quit(status=0)
  }
} else {
  cat(toJSON(list(error="mode must be 'unknown' or 'fixed'"), auto_unbox=TRUE))
  quit(status=0)
}

# Compile model to get tables (date_tab, RS_tab, FS_tab, b, etc.)
selected <- compile_model(selected_raw)

selected_text <- paste(capture.output(print(selected)), collapse = "\n")
full_text     <- paste(capture.output(print(res_all)), collapse = "\n")

out <- list(
  n = n,
  model = model,
  ic = ic,
  eps1 = eps1,
  max_breaks = max_breaks,
  mode = mode,
  k_fixed = k_fixed,

  breakpoints = as.integer(selected$b),
  date_tab = selected$date_tab,
  RS_tab = selected$RS_tab,
  FS_tab = selected$FS_tab,

  selected_model_text = selected_text,
  full_procedure_text = full_text
)

cat(toJSON(out, auto_unbox = TRUE, dataframe = "rows"))
