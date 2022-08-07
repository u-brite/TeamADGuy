### PreProcessing of Datasets
# ROSMAP
x = read.csv('/Users/ppugale/Desktop/ROSMAP_clinical.csv')

subj = read.csv('/Users/ppugale/Downloads/y_rosmap_whole_gene_expression.csv')
subj = colnames(subj)
subj_new = c()
for(i in subj){subj_new = c(subj_new,strsplit(i,'X')[[1]][2])}

rosmap = x[(as.character(subj_x) %in% subj_new),]

rosmap = rosmap[,c(1,4,8,10)]
rosmap
write.csv(rosmap,'/Users/ppugale/Desktop/ROSMAP_demographics.csv',quote = F)





# GSE63063
met1 = read.csv('/Users/ppugale/Downloads/GSE63063_series_matrix_EXPRESSIONdata.csv')
sub1 = read.csv('/Users/ppugale/Downloads/GSE63063_series_matrix_METAdata.csv',header = F)

sub1 = sub1[,c(1,3,11,13,14,43)]
colnames(sub1) = c('Subj','A','StatusA','Age','Gender','Status')
sub1 = sub1[-1,]
for(i in 1:nrow(sub1)){
  sub1$Age[i] = as.numeric(strsplit(sub1$Age[i],':')[[1]][2])
  sub1$Gender[i] = as.character(strsplit(sub1$Gender[i],':')[[1]][2])
}
head(sub1)
sub1$A = NULL
sub1$StatusA = NULL

subj_AD = sub1[sub1$Status == 'AD',]
subj_MCI = sub1[sub1$Status == 'MCI',]
subj_CTL = sub1[sub1$Status == 'CTL',]
subj_CTL_to_AD = sub1[sub1$Status == 'CTL to AD',]
subj_MCI_to_CTL = sub1[sub1$Status == 'MCI to CTL',]

met1_AD = met1[,subj_AD$Subj]
met1_MCI = met1[,subj_MCI$Subj]
met1_CTL = met1[,subj_CTL$Subj]
genes = met1$Probe
write.csv(genes,'/Users/ppugale/Desktop/GSE63063_genes.csv',quote = F,row.names=F)
write.csv(subj_AD,'/Users/ppugale/Desktop/GSE63063_AD_demo.csv',quote = F,row.names = F)
write.csv(subj_MCI,'/Users/ppugale/Desktop/GSE63063_MCI_demo.csv',quote = F,row.names = F)
write.csv(subj_CTL,'/Users/ppugale/Desktop/GSE63063_CTL_demo.csv',quote = F,row.names = F)

write.csv(met1_AD,'/Users/ppugale/Desktop/GSE63063_AD.csv',quote = F,row.names = F)
write.csv(met1_MCI,'/Users/ppugale/Desktop/GSE63063_MCI.csv',quote = F,row.names = F)
write.csv(met1_CTL,'/Users/ppugale/Desktop/GSE63063_CTL.csv',quote = F,row.names = F)



# miRNA data
miRNA = read.csv('/Users/ppugale/Downloads/GSE150693_series_matrix_EXPRESSIONdata_MCI2AD.csv')
subdata = read.csv('/Users/ppugale/Downloads/GSE150693_series_matrix_METAdata_MCI2AD.csv',header = F)
sub1 = subdata[,c(1,3,47,12,13)]
colnames(sub1) = c('Subj','A','Status','Age','Gender')
sub1 = sub1[-1,]
for(i in 1:nrow(sub1)){
  sub1$Age[i] = as.numeric(strsplit(sub1$Age[i],':')[[1]][2])
  #sub1$Status[i] = as.numeric(strsplit(sub1$Status[i],':')[[1]][2])
  sub1$Gender[i] = as.character(strsplit(sub1$Gender[i],':')[[1]][2])
}


miRNA_mci_ad = sub1[sub1$Status == 'MCI-C',]
miRNA_mci_ctl = sub1[sub1$Status == 'MCI-NC',]

miRNA_ad = miRNA[,miRNA_mci_ad$Subj]
miRNA_ctl = miRNA[,miRNA_mci_ctl$Subj]


write.csv(miRNA_mci_ad,'/Users/ppugale/Desktop/miRNA_AD_demo.csv',quote = F,row.names = F)
write.csv(miRNA_mci_ctl,'/Users/ppugale/Desktop/miRNA_CTL_demo.csv',quote = F,row.names = F)

write.csv(miRNA_ad,'/Users/ppugale/Desktop/miRNA_AD.csv',quote = F,row.names = F)
write.csv(miRNA_ctl,'/Users/ppugale/Desktop/miRNA_MCI.csv',quote = F,row.names = F)

