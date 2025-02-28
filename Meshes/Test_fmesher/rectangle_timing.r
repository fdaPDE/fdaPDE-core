library(fmesher)

# Rectangle
loc <- matrix(c(
  0.0, 0.0,
  4000.0, 0.0,
  4000.0, 2000.0,
  0.0, 2000.0
), ncol = 2, byrow = TRUE)

segments <- cbind(1:nrow(loc), c(2:nrow(loc), 1))
segm_obj <- fm_segm(loc = loc, idx = segments, closed = TRUE)
boundary <- fm_as_segm(segm_obj)

edge_values <- c(176, 61, 22.5, 18)

# Initialize a data frame to store results
results <- data.frame(
  NumPoints = numeric(),
  TimeElapsed_ms = numeric()
)

for (edge in edge_values) {
  cat("Generating mesh with max.edge =", edge, "\n")
  
  t0 <- Sys.time()
  mesh <- fm_mesh_2d(
    boundary = boundary,
    max.edge = edge,
    min.angle = 20,
    delaunay = TRUE
  )
  elapsed_ms <- as.numeric(difftime(Sys.time(), t0, units = "secs"))*1000
  num_points <- nrow(mesh$loc)
  cat("Number of points in mesh:", num_points, "\n")
  cat("Time elapsed (ms):", elapsed_ms, "\n\n")

  results <- rbind(results, data.frame(NumPoints = num_points, TimeElapsed_ms = elapsed_ms))
}

n <- results$NumPoints
t <- results$TimeElapsed_ms
n_min <- n[1]
t_min <- t[1]

curve_nlogn   <- t_min / (n_min * log(n_min)) * n * log(n)
curve_n       <- t_min / n_min * n
curve_n2      <- t_min / (n_min^2) * n^2
curve_n2logn  <- t_min / (n_min^2 * log(n_min)) * n^2 * log(n)

output_dir <- "/mnt/Meshes/Test_fmesher"
png(file.path(output_dir, "timing_plot_fmesher.png"), width = 800, height = 600)

plot(n, t, log = "xy", type = "b", col = "blue", pch = 16,
     xlab = "Number of Points", ylab = "Time (ms)",
     main = "Computational Time vs Number of Points")

lines(n, curve_nlogn, col = "red", lty = 2)
lines(n, curve_n,     col = "green", lty = 2)
lines(n, curve_n2,    col = "orange", lty = 2)
lines(n, curve_n2logn,col = "purple", lty = 2)

legend("topleft", legend = c("Measured Time", "O(n log n)", "O(n)", "O(n²)", "O(n² log n)"),
       col = c("blue", "red", "green", "orange", "purple"), lty = c(1,2,2,2,2), pch = c(16, NA, NA, NA, NA))

