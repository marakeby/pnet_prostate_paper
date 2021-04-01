##Bioconductor version 3.12 (BiocManager 1.30.10), R 4.0.4 (2021-02-15)
## Installing package(s) 'edgeR'


library(edgeR)
library(ggplot2)

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

##normalize
y_normalized <- DGEList(data_clean)
y_normalized <- calcNormFactors(y_normalized)
##design <- model.matrix(~group + batch + rin)


#### 

get_group <- function(gene){
  filename<- paste("/Users/haithamelmarakeby/PycharmProjects/pnet2/_database/prostate/processed/", gene, "_amps.csv", sep = "")
  group.df<-read.csv(filename, row.names=1,  header = TRUE)
  return (group.df)
}

run_GDE <-function(gene,pvalue.th){
  group.df<-get_group(gene)
  group <- group.df[,gene]
  response <-group.df[,"response"]

  design <- model.matrix(~group +response)
  #design <- model.matrix(~group )
  
  y_est <- estimateDisp(y_normalized, design)
  fit <- glmFit(y_est, design)
  lrt <- glmLRT(fit, coef = 2)
  n = nrow(data_clean)
  tops<- topTags(lrt, n)
  print(tops[gene,])
  
  boxplot(as.numeric(data_clean[gene, ]) ~ group, ylab=c(gene, "Expression"))
  
  #tops
  
  plot_volcano (tops, gene, pvalue.th)
}

pvalue.th <-0.001
gene <-"MDM4"
run_GDE(gene, pvalue.th )

pvalue.th <-0.001
gene <-"MDM4"
run_GDE(gene, pvalue.th )


gene = "NOTCH1"
run_GDE(gene, pvalue.th )


gene = "PDGFA"
run_GDE(gene, pvalue.th )


