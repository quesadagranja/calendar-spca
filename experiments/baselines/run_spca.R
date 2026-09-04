#!/usr/bin/env Rscript

# High-dimensional SPCA reference used in the Calendar-SPCA paper.
#
# The implementation follows the arrayspc alternating scheme of Zou et al.:
# orthogonal alpha updates alternate with componentwise soft-thresholding of
# X'X alpha. The regularization parameter is expressed as gamma times the
# maximum PCA zero-solution threshold.

suppressPackageStartupMessages(library(irlba))

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(name, default = NULL) {
  index <- which(args == name)
  if (length(index) == 0L) return(default)
  if (index[1] == length(args)) stop(paste("Missing value after", name))
  args[index[1] + 1L]
}

input_file <- get_arg("--input")
output_dir <- get_arg("--output-dir")
n <- as.integer(get_arg("--n"))
p <- as.integer(get_arg("--p"))
gamma <- as.numeric(get_arg("--gamma"))
K <- as.integer(get_arg("--components", "15"))
max_iter <- as.integer(get_arg("--max-iter", "200"))
eps_conv <- as.numeric(get_arg("--eps-conv", "0.001"))
seed <- as.integer(get_arg("--seed", "20260901"))

if (is.null(input_file) || is.null(output_dir)) {
  stop("Required: --input PATH --output-dir PATH --n N --p P --gamma VALUE")
}
if (!file.exists(input_file)) stop("Input file does not exist")
if (!is.finite(n) || n < 1L || !is.finite(p) || p < 1L) stop("n and p must be positive")
if (!is.finite(gamma) || gamma <= 0) stop("gamma must be positive")
if (!is.finite(K) || K < 1L || K > min(n, p)) stop("Invalid component count")
if (!is.finite(max_iter) || max_iter < 1L) stop("max_iter must be positive")
if (!is.finite(eps_conv) || eps_conv <= 0) stop("eps_conv must be positive")

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

soft_matrix <- function(a, threshold) sign(a) * pmax(abs(a) - threshold, 0)
normalize_columns <- function(b) {
  norms <- sqrt(colSums(b * b))
  norms[norms == 0] <- 1
  sweep(b, 2L, norms, FUN = "/")
}
convcheck <- function(beta1, beta2) {
  plus <- apply(abs(beta1 + beta2), 2L, max)
  minus <- apply(abs(beta1 - beta2), 2L, max)
  max(pmin(plus, minus))
}
xtx_times <- function(X, B) crossprod(X, X %*% B)

cat("Calendar-SPCA paper baseline: Zou SPCA\n")
cat(sprintf("Input shape: %d x %d\n", n, p))
cat(sprintf("K=%d, gamma=%.8g\n", K, gamma))

started <- proc.time()
connection <- file(input_file, open = "rb")
values <- readBin(
  connection,
  what = "double",
  n = as.double(n) * as.double(p),
  size = 8L,
  endian = "little"
)
close(connection)
if (length(values) != as.double(n) * as.double(p)) stop("Unexpected binary length")
X <- matrix(values, nrow = n, ncol = p, byrow = TRUE)
rm(values)
X <- sweep(X, 2L, colMeans(X), FUN = "-")
if (any(!is.finite(X))) stop("Input contains non-finite values")
total_ss <- sum(X * X)

set.seed(seed)
initial <- irlba::irlba(X, nv = K, nu = 0L, tol = 1e-8, maxit = 2000L)
alpha <- as.matrix(initial$v[, seq_len(K), drop = FALSE])
singular_values <- as.numeric(initial$d[seq_len(K)])

# For a PCA loading v_k, X'X v_k = sigma_k^2 v_k. The exact
# componentwise zero threshold is sigma_k^2 ||v_k||_inf.
tau_k <- singular_values^2 * apply(abs(alpha), 2L, max)
tau_ref <- max(tau_k)
threshold <- gamma * tau_ref

beta <- soft_matrix(xtx_times(X, alpha), threshold)
previous <- normalize_columns(beta)
outer <- 0L
difference <- Inf
history <- data.frame(
  outer = integer(),
  difference = numeric(),
  elapsed_seconds = numeric(),
  nnz_min = integer(),
  nnz_mean = numeric(),
  nnz_max = integer()
)

while (outer < max_iter && difference > eps_conv) {
  outer <- outer + 1L
  iteration_start <- proc.time()

  alpha_raw <- xtx_times(X, beta)
  decomposition <- svd(alpha_raw, nu = K, nv = K)
  alpha <- decomposition$u %*% t(decomposition$v)

  beta <- soft_matrix(xtx_times(X, alpha), threshold)
  normalized <- normalize_columns(beta)
  difference <- convcheck(normalized, previous)
  previous <- normalized

  nnz <- colSums(abs(beta) > 1e-10)
  elapsed <- unname((proc.time() - iteration_start)[["elapsed"]])
  history <- rbind(
    history,
    data.frame(
      outer = outer,
      difference = difference,
      elapsed_seconds = elapsed,
      nnz_min = min(nnz),
      nnz_mean = mean(nnz),
      nnz_max = max(nnz)
    )
  )
  cat(sprintf(
    "outer=%3d diff=%.6e nnz=%d/%.1f/%d\n",
    outer, difference, min(nnz), mean(nnz), max(nnz)
  ))
}

B <- normalize_columns(beta)
active <- sqrt(colSums(B * B)) > 1e-12
sparsity <- colMeans(abs(B) <= 1e-10)

if (any(active)) {
  qr_b <- qr(B[, active, drop = FALSE], tol = 1e-10, LAPACK = FALSE)
  rank_b <- qr_b$rank
  Q <- qr.Q(qr_b, complete = FALSE)[, seq_len(rank_b), drop = FALSE]
  projected <- X %*% Q
  common_ev <- sum(projected * projected) / total_ss
} else {
  rank_b <- 0L
  common_ev <- 0
}

elapsed_total <- unname((proc.time() - started)[["elapsed"]])
converged <- is.finite(difference) && difference <= eps_conv

writeBin(
  as.double(B),
  file.path(output_dir, "loadings_colmajor_f64.bin"),
  size = 8L,
  endian = "little"
)
write.table(
  history,
  file.path(output_dir, "history.csv"),
  sep = ",",
  row.names = FALSE,
  quote = FALSE
)
write.table(
  data.frame(
    component = seq_len(K),
    active = active,
    sparsity = sparsity
  ),
  file.path(output_dir, "components.csv"),
  sep = ",",
  row.names = FALSE,
  quote = FALSE
)
write.table(
  data.frame(
    gamma = gamma,
    tau_ref = tau_ref,
    threshold = threshold,
    converged = converged,
    outer_iterations = outer,
    final_difference = difference,
    effective_rank = sum(active),
    loading_rank = rank_b,
    mean_sparsity = if (any(active)) mean(sparsity[active]) else NaN,
    common_projection_ev = common_ev,
    elapsed_seconds = elapsed_total
  ),
  file.path(output_dir, "summary.csv"),
  sep = ",",
  row.names = FALSE,
  quote = FALSE
)

cat(sprintf("Converged: %s\n", converged))
cat(sprintf("Common projection EV: %.6f%%\n", 100 * common_ev))
cat(sprintf("Output: %s\n", normalizePath(output_dir)))
