#!/usr/bin/env Rscript
# This file is part of fdaPDE, a C++ library for physics-informed
# spatial and functional data analysis.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2 || length(args) > 3) {
    stop("usage: Rscript plot_simd_sweep.R summary.csv output_directory [stops.csv]")
}
summary_file <- normalizePath(args[1], mustWork = TRUE)
out_dir <- args[2]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
out_dir <- normalizePath(out_dir, mustWork = TRUE)
stop_file <- if (length(args) == 3) args[3] else file.path(dirname(summary_file), "stops.csv")
data <- read.csv(summary_file, stringsAsFactors = FALSE, check.names = FALSE)
required <- c("compiler", "suite", "case", "size", "comparison", "rows", "inner", "cols", "scalar",
              "coefficients", "off_ns", "on_ns", "ratio", "ratio_min", "ratio_max",
              "lhs_order", "rhs_order", "output_order")
if (!all(required %in% names(data)) || nrow(data) == 0) stop("summary CSV is empty or lacks required columns")
positive <- c("size", "rows", "cols", "coefficients", "off_ns", "on_ns", "ratio", "ratio_min", "ratio_max")
if (any(vapply(data[positive], function(x) any(!is.finite(x) | x <= 0), logical(1)))) {
    stop("dimensions, timings and ratios must be finite and positive")
}
if (any(data$ratio_min > data$ratio | data$ratio > data$ratio_max) ||
    any(!data$suite %in% c("assignment", "product"))) stop("invalid paired ranges or suite labels")
if (any(vapply(data[c("lhs_order", "rhs_order", "output_order")],
               function(x) any(!x %in% c(0, 1)), logical(1)))) stop("layout metadata must use 0=row, 1=column")
if (any(data$inner[data$suite == "product"] <= 0)) stop("product throughput requires positive inner dimensions")
if (!capabilities("cairo")) stop("the installed R runtime requires cairo support for SVG and PNG export")
stops <- if (file.exists(stop_file)) read.csv(stop_file, stringsAsFactors = FALSE) else data.frame()
data$short_round <- if ("short_round" %in% names(data)) tolower(as.character(data$short_round)) == "true" else FALSE
data$layout <- paste0(ifelse(data$lhs_order == 0, "R", "C"), ifelse(data$rhs_order == 0, "R", "C"),
                      ifelse(data$output_order == 0, "R", "C"))
data$off_ns_per_coefficient <- data$off_ns / data$coefficients
data$on_ns_per_coefficient <- data$on_ns / data$coefficients
data$off_gflops <- data$on_gflops <- NA_real_
products <- data$suite == "product"
data$off_gflops[products] <- 2 * data$rows[products] * data$inner[products] * data$cols[products] / data$off_ns[products]
data$on_gflops[products] <- 2 * data$rows[products] * data$inner[products] * data$cols[products] / data$on_ns[products]
write.csv(data, file.path(out_dir, "plot_data.csv"), row.names = FALSE)

# keep the paired ratio separate from the ratio of independently summarized absolute times
comparisons <- c("off:assignment" = "A0P0 / A1P0", "assignment:all" = "A1P0 / A1P1",
                 "off:product" = "A0P0 / A0P1", "off:all" = "A0P0 / A1P1")
colors <- c("off:assignment" = "#0072B2", "assignment:all" = "#0072B2",
            "off:product" = "#D55E00", "off:all" = "#009E73")
mode_labels <- c("off" = "A0P0", "assignment" = "A1P0", "product" = "A0P1", "all" = "A1P1")

# report the recorded endpoint reason without inferring a plateau from an incomplete sweep
stop_caption <- function(d) {
    needed <- c("compiler", "suite", "case", "reason", "plateau_observed")
    if (!all(needed %in% names(stops))) return("stop metadata unavailable; no plateau claim")
    found <- stops[stops$compiler == d$compiler[1] & stops$suite == d$suite[1] & stops$case == d$case[1], , drop = FALSE]
    if (nrow(found) == 0) return("sweep in progress; no recorded plateau claim")
    found <- found[nrow(found), , drop = FALSE]
    plateau <- tolower(as.character(found$plateau_observed)) == "true"
    paste("stop:", gsub("_", " ", found$reason), "|",
          if (plateau) "local plateau confirmed" else "local plateau not observed")
}

# show exact endpoint shapes because input size N need not equal every product dimension
shape_caption <- function(d) {
    endpoint <- d[c(1, nrow(d)), , drop = FALSE]
    if (d$suite[1] == "assignment") {
        shapes <- paste(endpoint$rows, endpoint$cols, sep = " x ")
        return(paste("shape:", paste(unique(shapes), collapse = " to "), "|", d$scalar[1]))
    }
    shapes <- sprintf("N=%s: %s x %s times %s x %s -> %s x %s", endpoint$size, endpoint$rows,
                      endpoint$inner, endpoint$inner, endpoint$cols, endpoint$rows, endpoint$cols)
    paste(unique(shapes), collapse = "; ")
}

# log axes retain small-size overhead and memory-regime behavior on the same page
metric_panel <- function(x, off, on, x_label, y_label, legend_labels) {
    plot(x, off, type = "n", log = "xy", ylim = range(c(off, on)) * c(0.9, 1.1),
         xlab = x_label, ylab = y_label, las = 1)
    grid(col = "#e4e4e4")
    lines(x, off, type = "o", pch = 16, col = "#444444", lwd = 1.5)
    lines(x, on, type = "o", pch = 17, col = "#0072B2", lwd = 1.5)
    legend("top", legend_labels, inset = c(0, -0.15), xpd = NA, horiz = TRUE, col = c("#444444", "#0072B2"), pch = c(16, 17), lty = 1,
           bty = "n", cex = 0.8)
}

# ratios use process-pair min/max segments; secondary factorial comparisons remain discrete anchors
draw_page <- function(d) {
    d <- d[order(d$size, d$comparison), , drop = FALSE]
    assignment <- d$suite[1] == "assignment"
    primary_name <- if (assignment) "off:assignment" else "assignment:all"
    primary <- d[d$comparison == primary_name, , drop = FALSE]
    if (nrow(primary) == 0) stop("a plotted workload lacks its primary comparison")
    x <- if (assignment) primary$coefficients else primary$size
    x_label <- if (assignment) "Coefficients" else "Input size N (exact shapes above)"
    par(mfrow = c(3, 1), mar = c(4.0, 5.0, 2.0, 1.0), oma = c(4.0, 0.0, 4.0, 0.0), cex = 0.9)
    all_x <- if (assignment) d$coefficients else d$size
    plot(all_x, d$ratio, type = "n", log = "xy", ylim = range(c(1, d$ratio_min, d$ratio_max)) * c(0.9, 1.1),
         xlab = x_label, ylab = "Paired OFF / ON ratio", las = 1)
    grid(col = "#e4e4e4")
    abline(h = 1, col = "#777777", lty = 2)
    present <- unique(c(primary_name, d$comparison))
    for (comparison in present) {
        curve <- d[d$comparison == comparison, , drop = FALSE]
        at <- if (assignment) curve$coefficients else curve$size
        color <- colors[[comparison]]
        if (is.null(color)) stop("unknown comparison label")
        segments(at, curve$ratio_min, at, curve$ratio_max, col = adjustcolor(color, alpha.f = 0.45), lwd = 4)
        if (comparison == primary_name) lines(at, curve$ratio, col = color, lwd = 1.5)
        points(at, curve$ratio, col = color, pch = ifelse(curve$short_round, 1, 16))
    }
    legend("top", unname(comparisons[present]), inset = c(0, -0.15), xpd = NA, horiz = TRUE, col = unname(colors[present]), pch = 16,
           lty = ifelse(present == primary_name, 1, NA), bty = "n", cex = 0.8)
    labels <- unname(mode_labels[strsplit(primary_name, ":", fixed = TRUE)[[1]]])
    metric_panel(x, primary$off_ns, primary$on_ns, x_label, "Nanoseconds / public call", labels)
    if (assignment) {
        metric_panel(x, primary$off_ns_per_coefficient, primary$on_ns_per_coefficient, x_label,
                     "Nanoseconds / coefficient", labels)
    } else {
        metric_panel(x, primary$off_gflops, primary$on_gflops, x_label,
                     "Logical GFLOP/s", labels)
    }
    mtext(paste(gsub("_", " ", d$case[1]), "|", d$scalar[1], "|", d$compiler[1]), outer = TRUE,
          side = 3, line = 2.7, font = 2, cex = 1.0)
    caption <- shape_caption(primary)
    if (!assignment) caption <- paste("lhs/rhs/output:", d$layout[1], "|", caption)
    mtext(paste(strwrap(caption, width = 100), collapse = "\n"), outer = TRUE, side = 3, line = 0.5, cex = 0.75)
    pairs <- if ("pair_count" %in% names(d)) paste(unique(d$pair_count), collapse = "/") else "reported"
    note <- paste("median paired ratios; segments: min-max across", pairs, "process pairs; no confidence interval")
    if (any(d$short_round)) note <- paste(note, "| open markers: round below 1 ms")
    if (!assignment) note <- paste(note, "| logical FLOPs=2mnk; times retain public-call overhead")
    mtext(paste(strwrap(note, width = 110), collapse = "\n"), outer = TRUE, side = 1, line = 1.4, cex = 0.7)
    mtext(stop_caption(d), outer = TRUE, side = 1, line = 0.1, cex = 0.7)
}

# group assignment PDFs by operation and product PDFs by scalar and physical operand/output layouts
operation <- sub("^.*_", "", data$case)
data$bucket <- ifelse(data$suite == "assignment", paste0("assignment_", operation),
                      paste0("product_", data$scalar, "_", data$layout))
page_key <- interaction(data$compiler, data$case, data$scalar, drop = TRUE)
index <- data.frame(file = character(), format = character(), pages = integer(), stringsAsFactors = FALSE)
for (bucket in unique(data$bucket)) {
    selected <- data[data$bucket == bucket, , drop = FALSE]
    pages <- split(selected, page_key[data$bucket == bucket], drop = TRUE)
    pdf_path <- file.path(out_dir, paste0(bucket, ".pdf"))
    pdf(pdf_path, width = 8, height = 9, onefile = TRUE)
    for (page in pages) draw_page(page)
    dev.off()
    index <- rbind(index, data.frame(file = basename(pdf_path), format = "pdf", pages = length(pages)))
    for (page in pages) {
        name <- paste(bucket, page$case[1], page$compiler[1], sep = "-")
        name <- gsub("[^A-Za-z0-9_.-]", "_", name)
        svg_path <- file.path(out_dir, paste0(name, ".svg"))
        svg(svg_path, width = 8, height = 9)
        draw_page(page)
        dev.off()
        png_path <- file.path(out_dir, paste0(name, ".png"))
        png(png_path, width = 8, height = 9, units = "in", res = 180, type = "cairo")
        draw_page(page)
        dev.off()
        index <- rbind(index, data.frame(file = c(basename(svg_path), basename(png_path)),
                                         format = c("svg", "png"), pages = 1L))
    }
}
write.csv(index, file.path(out_dir, "plots.csv"), row.names = FALSE)
cat(sprintf("wrote %d standalone graphics to %s\n", nrow(index), out_dir))
