##Bioconductor version 3.12 (BiocManager 1.30.10), R 4.0.4 (2021-02-15)
## Installing package(s) 'edgeR'

library(edgeR)
library(ggplot2)
#read counts
data_raw <- read.csv("/Users/haithamelmarakeby/PycharmProjects/pnet2/_database/prostate/processed/p1000_read_counts.csv", row.names=1,  header = TRUE)
dim(data_raw)
head(data_raw)


## clean data
cpm_log <- cpm(data_raw, log = TRUE)
median_log2_cpm <- apply(cpm_log, 1, median)
hist(median_log2_cpm)
expr_cutoff <- -1
abline(v = expr_cutoff, col = "red", lwd = 3)
sum(median_log2_cpm > expr_cutoff)
data_clean <- data_raw[median_log2_cpm > expr_cutoff, ]
dim(data_clean)
#################

##cpm_log <- cpm(data_clean, log = TRUE)
##heatmap(cor(cpm_log))

# get group data 
filename<- "/Users/haithamelmarakeby/PycharmProjects/pnet2/_database/prostate/processed/MDM4_amps.csv"
##filename <-"/Users/haithamelmarakeby/PycharmProjects/pnet2/_database/prostate/processed/NOTCH1_amps.csv"
##filename <-"/Users/haithamelmarakeby/PycharmProjects/pnet2/_database/prostate/processed/PDGFA_amps.csv"
##gene = "NOTCH1"
##gene = "PDGFA"
gene = "MDM4"
group.df<-read.csv(filename, row.names=1,  header = TRUE)
group <- group.df[,gene]
response <-group.df[,"response"]

n = nrow(data_clean)
y <- DGEList(counts = data_clean, group = group)
y_normalized <- calcNormFactors(y)
head(y$samples)

y_est <- estimateDisp(y_normalized)

#sqrt(y_est$common.dispersion) # biological coefficient of variation
#plotBCV(y_est)

et <- exactTest(y_est)
results_edgeR <- topTags(et, n = n, sort.by = "PValue")
head(results_edgeR$table)

# plot high genes
pvalue.th <-0.001
sum(results_edgeR$table$FDR < pvalue.th)
plotSmear(et, de.tags = rownames(results_edgeR)[results_edgeR$table$FDR < pvalue.th])
plotSmear(et, de.tags = c(gene))
abline(h = c(-0.9, 0.9), col = "blue")

print(results_edgeR[gene,])
rownames(results_edgeR)[results_edgeR$table$FDR < pvalue.th]


plot_volcano <- function(results_edgeR, gene, threshold)
{
results_edgeR.df <-as.data.frame(results_edgeR)
results_edgeR.df$log10FDR <- -log10(results_edgeR.df$FDR)
results_edgeR.df$threshold = as.factor(results_edgeR.df$FDR < threshold)

head(results_edgeR.df)

g <- ggplot(data=results_edgeR.df, 
            aes(x=logFC, y =log10FDR, 
                colour=threshold)) +
  geom_point(alpha=0.4, size=1) +
  ##xlim(c(-14, 14)) +
  xlab("log2 fold change") + ylab("-log10 FDR") +
  theme_bw() +
  scale_color_manual(values = c("blue", "red")) +
  theme(legend.position="none")+
  geom_label(data=results_edgeR.df[gene,], aes(label=gene))
g 
}
  




##group <- mdm4_amps$MDM4




y <- DGEList(data_clean)
y <- calcNormFactors(y)
##design <- model.matrix(~group + batch + rin)
design <- model.matrix(~group +response)


design

y <- estimateDisp(y, design)
fit <- glmFit(y, design)
lrt <- glmLRT(fit, coef = 2)
n = nrow(data_clean)
tops<- topTags(lrt, n)
tops[gene,]

boxplot(as.numeric(data_clean[gene, ]) ~ group, ylab=c(gene, "Expression"))

#tops

plot_volcano (tops, gene, 0.001)


x = results_edgeR.df["MDM4","logFC"]
y = results_edgeR.df["MDM4","FDR"]
