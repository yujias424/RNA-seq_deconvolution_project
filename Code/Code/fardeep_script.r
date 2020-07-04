library(FARDEEP)

mixture_1 <- read.csv('/ua/shi235/IndependentStudy/Data/tempdata/mixture_fardeep.txt',sep = '\t', row.names = 1)
LM22_1 <- read.csv('/ua/shi235/IndependentStudy/Data/tempdata/fardeep_signature.tsv',sep = '\t', row.names = 1)

rownames(mixture_1) <- NULL
rownames(LM22_1) <- NULL
mixture_1$x1 <- mixture_1$X0

result = fardeep(LM22_1, mixture_1)
coef = result$relative.beta

write.csv(x = coef, file = '/ua/shi235/IndependentStudy/Data/tempdata/fardeep_result.csv')
