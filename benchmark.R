cat("Making Graphs: ")

args <- commandArgs(trailingOnly = TRUE)
jobname = "plantdataL_1"
jobname = args[1]
jobfilename = args[2]

directory = paste0(jobfilename, "_results/" , jobname, "/")

# benchmarking
wid = 1500; hei = 2500
alloutputs <- read.csv(paste(directory, "alloutputs.", jobname , ".tsv", sep = ""), sep = "\t")
#alloutputs$RandomDataDiff = abs(alloutputs$RandomData - alloutputs$Actual)
#alloutputs$Benchmark_Adiff = abs(alloutputs$Benchmark_A - alloutputs$Actual)
#alloutputs$Benchmark_Bdiff = abs(alloutputs$Benchmark_B - alloutputs$Actual)

datasets <- list(alloutputs$BenchmarkAdiff.NNWdiff, alloutputs$BenchmarkBdiff.NNWdiff, alloutputs$RandomDatadiff.NNWdiff)
name <- list("fly_rnaiPredictor", "Deep_SP_Cas9", "RandomGuesses")
for(i in 1:3) {
  cat (i)
  data = datasets[[i]]
  png(filename = paste(directory, "Benchmarkto_", name[[i]], jobname, ".jpg",sep = ""), width = wid, height = hei,res=300)
  hist(datasets[[i]], col = "antiquewhite4",xlim= c(-1, 1), ylim = c(0, length(data)/2), 
       main = paste0("Benchmark NNW to ", name[[i]]),xlab = paste0("Difference between NNW error and ", name[[i]], " error"))
  col = "red2"
  if(mean(data) > 0) col = "forestgreen"
  abline(v = mean(data), col = col, lwd = 2)
  mtext(side = 3, paste0("on average, the NNW predicts ",  format(mean(data), digits = 4), " closer"))
  dev.off()
}

# comparing stats

cat('4')

#data <- data.frame(diffs$V1, randomdiffs$V1)
data <- data.frame(alloutputs$NNWDiff, alloutputs$RandomDatadiff, alloutputs$Benchmark_Adiff, alloutputs$Benchmark_Bdiff)
df1_summary<-as.data.frame(apply(data,2,summary))
#trainingerror<-read.csv(paste("finalavglossPERsubset.dat.",append, sep = ""))
#trainingerrorsummary<-as.data.frame(apply(trainingerror,2,summary))
wid = 1500; hei =2500
png(filename = paste(directory, "datasummary.", jobname, ".png",sep = ""), width = wid, height = hei,res=300)
plot(main = paste("Statistics for",length(data$alloutputs.Diff),"tests"), df1_summary$alloutputs.NNWDiff, axes = FALSE, 
     ylab = "Deviation from actual Effectiveness", xlab = "Statistic", ylim = c(0,1), pch = 20)
points(df1_summary$alloutputs.RandomDataDiff, pch = 2)
points(df1_summary$alloutputs.Benchmark_Adiff, pch = 9)
points(df1_summary$alloutputs.Benchmark_Bdiff, pch = 10)
legend("topleft",legend = c(
  paste("Neural Network, ", "\U{0078}\U{0304}","=" , format(df1_summary$alloutputs.NNWDiff[4], digits = 4)), 
  paste("Random Outputs, ", "x̄","=" , format(df1_summary$alloutputs.RandomDatadiff[4], digits= 4 )),
  paste("Benchmark A, ", "\U{0078}\U{0304}","=" , format(df1_summary$alloutputs.Benchmark_Adiff[4], digits = 4)),
  paste("Benchmark B, ", "\U{0078}\U{0304}","=" , format(df1_summary$alloutputs.Benchmark_Bdiff[4], digits = 4)),
  paste("Prediction Score:" , format((df1_summary$alloutputs.RandomDatadiff[4] / df1_summary$alloutputs.NNWDiff[4]), digits = 4)) ) , 
pch = c(20, 2, 9, 10, 0, 0) )
legend("bottomright", legend = c(paste("P Value:", format(
  t.test(alloutputs$NNWDiff, alloutputs$RandomDatadiff, alternative = "l")$p.value, digits = 3) )))
axis(1, at = 1:6, labels =  c("Min", "1st Qt", "Median", "Mean", "3rd Qt", "Max"), cex.axis = 0.8)
axis(2)
invisible(dev.off())

# violin graph, density graph
library(ggplot2)

combined_df <- rbind(data.frame(value = alloutputs$NNWDiff, Predictor = rep("nnw", length(alloutputs$NNWDiff))),
                      data.frame(value = alloutputs$RandomDatadiff, Predictor = rep("random", length(alloutputs$RandomDatadiff))), 
                     data.frame(value = alloutputs$Benchmark_Adiff, Predictor = rep("benchmA", length(alloutputs$Benchmark_Adiff))),
                     data.frame(value = alloutputs$Benchmark_Bdiff, Predictor = rep("benchmB", length(alloutputs$Benchmark_Bdiff))))

customcolors <- c("nnw" = "springgreen4", "random" = "indianred3", "benchmA" = "mediumorchid4", "benchmB" = "mediumpurple4", "actual" = "black")
pvalues <- c(format(t.test(alloutputs$NNWDiff, alloutputs$RandomDatadiff, alternative = "l")$p.value, digits = 3),
             format(t.test(alloutputs$NNWDiff, alloutputs$Benchmark_Adiff, alternative = "l")$p.value, digits = 3),
             format(t.test(alloutputs$NNWDiff, alloutputs$Benchmark_Bdiff, alternative = "l")$p.value, digits = 3),
             "P-Values")
cat("5")
totaltests <- length(alloutputs$NNWDiff)
png(filename = paste(directory, "error_density", jobname, ".png",sep = ""), width = wid, height = hei,res=300)
ggplot(combined_df, aes(x = value, color = Predictor)) + geom_density(linewidth = 1.5) + scale_color_manual(values = customcolors) +
  labs(x = "Error", y = "Density", title = "Deviation from Expected (error) Distribution", subtitle = bquote(bold(.(jobname)) ~ Total ~ .(totaltests) ~tests)  )
invisible(dev.off())

cat("6")
png(filename = paste(directory, "violin", jobname, ".png",sep = ""), width = wid, height = hei,res=300)
ggplot(combined_df, aes(x = Predictor, y = value, fill = Predictor)) + geom_violin(trim = TRUE, fill = "white") + geom_boxplot(width = 0.1) + stat_summary(fun = "mean", geom="point", shape=23, size=2) + 
  labs(y = "Deviation from Expected (error)", title = "Deviation from Expected Densities", subtitle = bquote(bold(.(jobname)) ~ Total ~ .(totaltests) ~tests) ) + scale_fill_manual(values = customcolors) +
  annotate("text", x = c("random", "benchmA", "benchmB", "nnw"), y = -0.05, label = format(pvalues, nsmall = 2))
invisible(dev.off())


# distribution density graph
combined_df <- rbind(data.frame(value = alloutputs$NNW, Predictor = rep("nnw", totaltests)),
                     data.frame(value = alloutputs$RandomData, Predictor = rep("random", totaltests)), 
                     data.frame(value = alloutputs$Benchmark_A, Predictor = rep("benchmA", totaltests)),
                     data.frame(value = alloutputs$Benchmark_B, Predictor = rep("benchmB", totaltests)))

cat("7")
png(filename = paste(directory, "prediction_density", jobname, ".png", sep = ""), width = wid, height = hei,res=300)
ggplot(combined_df, aes(x = value, color = Predictor)) + geom_density(linewidth = 1.5) + scale_color_manual(values = customcolors) +
  labs(x = "Output", y = "Density", title = "Distribution of Predictions", subtitle = bquote(bold(.(jobname)) ~ Total ~ .(totaltests) ~tests) )
invisible(dev.off())


# correlation graph
cat("8")
new_df <- data.frame(Actual = as.numeric(alloutputs$Actual), NNW = as.numeric(alloutputs$NNW))
png(filename = paste(directory, "correlation_test", jobname, ".png", sep = ""), width = wid, height = hei,res=300)
ggplot(new_df, aes(x = NNW , y = Actual)) + stat_density_2d(geom = "raster", aes(fill = after_stat(density)), contour = FALSE) + scale_fill_gradient(low = "white", high = "black") +
  labs(title = "Correlation Test", subtitle = bquote(bold(.(jobname)) ~ Total ~ .(totaltests) ~tests) ) + theme_bw()
invisible(dev.off())


cat ("_DONE")

