# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # Numpy
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# %%
data = pd.read_csv(
    r'/Users/priya/Documents/Comp_Bio/GitHub/Module-4-Cancer/data/TRAINING_SET_GSE62944_subsample_log2TPM.csv', index_col=0, header=0)  # can also use larger dataset with more genes
metadata_df = pd.read_csv(
    r'/Users/priya/Documents/Comp_Bio/GitHub/Module-4-Cancer/data/TRAINING_SET_GSE62944_metadata.csv', index_col=0, header=0)
cancer_type = 'SKCM'  # Skin Cutaneous Melanoma
# From metadata, get the rows where "cancer_type" is equal to the specified cancer type
# Then grab the index of this subset (these are the sample IDs)
cancer_samples = metadata_df[metadata_df['cancer_type'] == cancer_type].index
print(len(cancer_samples))
# %%
# Subset the main data to include only these samples
# When you want a subset of columns, you can pass a list of column names to the data frame in []
SKCM_data = data[cancer_samples]
#this list of genes is from both hallmarks we are concerned with, it has been cleaned to remove genes that are not in the dataset and any duplicates between the hallmarks 
desired_gene_list = [
    "ANGPTL3", "CXCR4", "LAMA4", "EFNB1", "MAP3K8", "RHOA", "TBL1XR1", "WNT5B",
    "LAMA5", "ITGA2B", "SDC1", "NGF", "NFKB2", "NTNG1", "HCK", "PDPK1", "CDH5",
    "WNT10A", "MAPKAPK2", "PIP5K1C", "CDC42BPA", "ABL2", "RAB11A", "TGFA", "ARRB1",
    "ITGA5", "PLCG1", "RASA1", "RRAS", "ENG", "CLDN15", "PIK3CD", "LAMB2", "CNR1",
    "WAS", "CLDN20", "FPR2", "PARD3", "SDC3", "ELMO2", "GP1BA", "FYN", "LIMK1",
    "PXN", "MAP3K2", "XIAP", "ERBB3", "CD6", "NRXN2", "PRKACB", "GFRA1", "TNF",
    "PRKACA", "S1PR1", "ITGB1", "PLXND1", "DUSP1", "BAMBI", "COL6A3", "NTF3",
    "MMP7", "CDC42", "RAPGEF1", "PTEN", "FGF10", "CD34", "FLNA", "ALCAM", "CRKL",
    "LAMC1", "AKT1", "IRS2", "SMAD2", "CAV1", "MADCAM1", "BMP7", "ACTR2", "CD44",
    "CLDN3", "TLR2", "ID1", "GNG2", "LAMB1", "GNA12", "ITGAL", "EFNA3", "NCKAP1",
    "SPHK1", "WNT9A", "SSX2IP", "DPP4", "NLGN2", "CTNNA1", "PRKCD", "PLCB3",
    "EPHB2", "CASP8", "TRIP6", "FGFR1", "EFNA1", "GNB5", "MYH9", "EPHB1", "ITGB2",
    "CBL", "CNTNAP1", "FAT1", "NRXN1", "ARHGEF7", "CD99", "NTRK2", "JUP", "NCAM1",
    "CDH4", "F11R", "RB1", "COL1A1", "EPHA2", "FN1", "IGF1R", "MYLPF", "PARVA",
    "EPHA3", "PIK3CB", "SNAI1", "WNT7A", "WNT3A", "NFASC", "SELPLG", "DDX5",
    "ITGAM", "GNAO1", "TJP1", "PARVG", "EPHA4", "ACTN1", "MYC", "CXCR3", "MAPK7",
    "ACVR1", "ARF6", "RAB5A", "ICAM3", "CD22", "EXOC2", "CLDN1", "ACVR2B",
    "CLDN10", "RPS6KB1", "CLDN19", "NTRK1", "PIK3CA", "CD36", "MAPK8IP3", "LPAR1",
    "SIGLEC1", "GPC4", "LAMB3", "GAB1", "NCK2", "MAP4K1", "WNT2", "WASF2", "CD86",
    "CADM3", "FER", "MAPK14", "PTK2", "CTTN", "STAT5B", "DVL1", "CADM1", "TLR4",
    "MTOR", "ARF4", "F2RL3", "CXCL12", "ARHGDIA", "WASL", "CLDN9", "PLCB1",
    "NRCAM", "PLD2", "LCP2", "DOCK2", "PRKCE", "KDR", "PDGFB", "MMP2", "COMP",
    "MSN", "FZD2", "PRKCG", "SYMPK", "RAF1", "JUN", "KIT", "EGFR", "VAV2", "SMPD1",
    "MAP3K7", "JAM2", "APC", "CASP3", "ARRB2", "NRG2", "ERBB2", "ACTN4", "FGF1",
    "CD63", "SOS1", "GNB1", "PRKCQ", "GPC3", "HGF", "MAP3K3", "CD47", "MAP2K6",
    "NLGN3", "CLDN16", "IBSP", "CD28", "SHC2", "NFATC2", "PAK2", "PVRL2", "EPHA1",
    "ITGB7", "FLT1", "TLN1", "ARHGEF1", "MAP4K3", "PPP3R1", "RALA", "ERBB4",
    "MAPRE1", "TAOK2", "MAPKAPK5", "SMAD3", "MMP3", "NEO1", "SIPA1", "ITGAE",
    "BDNF", "CLDN18", "RAP1B", "PRKG1", "ITGB5", "GNAI3", "PFN1", "LPAR3", "SRC",
    "CASK", "PTPRJ", "WNT5A", "ZAP70", "CD2AP", "IKBKB", "PAK1", "CLDN23", "NTRK3",
    "DRD2", "PARD6A", "MAPK12", "PKN2", "ITGA7", "RAC2", "PPARD", "ELMO1", "ITGA8",
    "FGFR4", "THBS1", "KITLG", "BRAF", "RND2", "PTPRF", "THBS2", "VAV3", "ITGA4",
    "RASGRF1", "EZR", "LAMA2", "FGFR2", "TNXB", "EFNB2", "PTK6", "CDH2", "EGF",
    "NRG1", "ACVRL1", "RND3", "VASP", "ESR1", "PLEKHA7", "PRICKLE1", "RAC1", "NF1",
    "EPHB4", "GRB2", "BAIAP2", "RRAS2", "WNT11", "WASF1", "IL1A", "EREG",
    "ARHGAP35", "PRKCZ", "FLT4", "RALGDS", "PTPN6", "COL6A6", "P2RY1", "HPSE",
    "LPAR2", "MYL9", "PTPRM", "YES1", "SHROOM2", "NGFR", "ETS1", "ACTB", "PTPN1",
    "NCK1", "ARPC5", "ARAP3", "MDK", "THBS3", "ITGA3", "PTPN11", "CLDN8", "PTK2B",
    "MAP2K2", "VTN", "NODAL", "ITGA9", "CLDN4", "PIP5K1A", "GPC2", "TGFBR2",
    "MAPKAPK3", "BCL2L1", "EPHB3", "ABI2", "MAX", "APC2", "CLDN11", "PRKCA",
    "SELL", "CCND1", "SDC4", "WASF3", "FAT4", "TNC", "BMPR2", "MAP2K4", "PDGFA",
    "MAP3K12", "RHOC", "BCL2", "PIK3R1", "SPN", "BMPR1A", "CDKN1A", "TLN2",
    "LAMC3", "ABL1", "PTPRC", "THY1", "EXOC3", "VCAN", "SHC3", "PVR", "GNAI1",
    "TGFB1", "ITGA6", "CASP9", "PRKD2", "CFL1", "ITGB4", "AKT2", "ITGA1",
    "PDGFRA", "NFKB1", "MAP2K3", "PRKCH", "HSPB1", "RET", "CRK", "CDK1", "MAPK3",
    "DOCK1", "PDGFRB", "ESAM", "DVL3", "PRKCI", "LAMA3", "VAV1", "TNFRSF1A",
    "GRIN2A", "FAS", "IGF1", "MMP14", "ROCK2", "COL6A1", "ACVR2A", "VWF", "JAM3",
    "PLCB2", "PLAU", "TRAF6", "DAAM1", "WNT4", "L1CAM", "SMAD4", "LAT", "ITGAV",
    "FES", "CDH1", "CD2", "DICER1", "SPP1", "LAMC2", "ZYX", "CLDN6", "CLDN2",
    "CCL11", "BMP2", "MMP9", "MAPK8IP2", "IQGAP1", "PSEN1", "HRAS", "PLAUR",
    "UNC5C", "NLGN4X", "ANGPT2", "TIAM1", "WNT3", "WNT7B", "LEF1", "SNAI2",
    "NTNG2", "KSR1", "MAGI1", "MAP4K4", "FARP2", "MAG", "JUND", "ITGA2", "TGFB2",
    "STAT3", "CREBBP", "NEGR1", "DRD1", "F2R", "PPP2R2A", "CDKN1B", "MYLK",
    "HLA-DRB1", "CDK5", "NOG", "ITGB8", "HIF1A", "RAP1A", "LAMA1", "CAMK2G",
    "GNAI2", "OMG", "TGFBR1", "RDX", "CAPN2", "GSK3B", "CTNNA2", "CD4", "WNT2B",
    "FGF2", "TCF7L2", "NOS3", "ILK", "MAP3K4", "CNTNAP2", "CLDN14", "NRAS",
    "SELP", "GJA1", "VEGFA", "RELN", "PPP3CA", "CDH11", "CD226", "KRAS", "GNAS",
    "WNT10B", "ROCK1", "PARVB", "EFNA5", "AJUBA", "FPR1", "EFNA2", "NRXN3", "FAP",
    "RHOB", "MYH10", "MAPK1", "MAPK8", "FGB", "HLA-DRA", "GPC1", "CDH15", "PLD1",
    "DAB1", "CNTN2", "CTNND1", "MET", "TP53", "LIMA1", "ITGA11", "SDC2", "NLGN1",
    "ANGPT1", "FRS2", "RIN1", "DNM2", "PIK3CG", "MAP3K11", "MAPK13", "DVL2",
    "SELE", "CLDN7", "ENAH", "DAXX", "SHC1", "NCAM2", "THBS4", "VCAM1", "MAPK10",
    "BSG", "ACTR3", "COL4A6", "ITGB3", "PAK4", "AKT3", "MAPK9", "DDR2", "PTPN13",
    "DIAPH1", "PPP2R1A", "PTPRB", "RALB", "PAFAH1B1", "BRK1", "MAP2K1", "ACTN2",
    "ICAM2", "CSF1", "MAP4K2", "COL1A2", "PPP2CA", "MAP3K1", "BMX", "CTNNB1",
    "TEK", "ABI1", "MAP2K7", "ARHGAP5", "GAB2", "FASLG", "BIRC3", "EXOC4", "BCAR1",
    "ITGA10", "ITGB6", "COL4A3", "CD80", "VCL", "MFGE8", "CSF3R", "TRAF2",
    "SLC9A1", "MAPK11", "PRKCB", "GNA11", "ICAM1", "CLDN5", "CBLB", "TGFB3",
    "WNT6", "NOTCH2", "MAP3K5", "COL6A2", "MAP2K5", "CNTN1", "CDH3", "HBEGF",
    "CYFIP2", "GNAQ", "STAT1", "RIPK1", "CDC25B", "INPP5D", "ACER2", "PPARG",
    "PPP1R12A", "GLI2", "CSNK1A1", "CACNA1I", "RPTOR", "BMP4", "FLT3", "CD40LG",
    "CSNK1D", "IL17D", "PPP1R12B", "DGKI", "PTCH1", "SMAD1", "SFN", "RASAL2",
    "CSF3", "PPP2R5E", "SERPINE1", "HDAC1", "MPL", "YWHAB", "TSLP", "BIRC2",
    "NSMAF", "CSNK1G2", "LPAR6", "RALBP1", "S1PR2", "PKN1", "STK3", "SH2D2A",
    "OSM", "NPM1", "AR", "GNB2", "GNG10", "FGF17", "SFRP5", "HDAC2", "CCND3",
    "PLCD3", "RASSF1", "RPS6KB2", "EPO", "SOCS3", "PHLPP2", "IRS1", "PTPN5",
    "IL6R", "BIRC5", "SLC44A4", "PGF", "SUZ12", "STK11", "ATF6B", "BCL2L11",
    "CSF2RB", "IHH", "DLG1", "IL2RG", "PLCD4", "SDCBP", "PRKAA1", "CSNK1G3", "IL7",
    "YWHAE", "MRAS", "PPP3CB", "RASGRP1", "EZH2", "RASAL1", "IL6ST", "HTR7", "GLI1",
    "LRP5", "SUFU", "ASAH1", "IL7R", "SHOC2", "JAK1", "GNG13", "GNB3", "RASSF6",
    "SGMS1", "YWHAG", "IL2RB", "CREB1", "NTF4", "RASA2", "MED1", "CBLC", "RHEB",
    "RASAL3", "SYK", "STAM2", "PLA2G1B", "FOS", "ADORA3", "FGF11", "TRAF1",
    "FGF13", "RIPK2", "HCLS1", "DLG4", "PPP1R12C", "HSP90AA1", "BRCA1", "FZD3",
    "FZD10", "VEGFB", "GATA3", "BDKRB2", "IL1R1", "CALML5", "SPDYA", "IL1B", "ID4",
    "RGL1", "CHRM1", "PLCE1", "LYN", "PIK3AP1", "ACVR1B", "CCNB1", "IGF2", "NF2",
    "FGF22", "MLST8", "RAPGEF2", "SPHK2", "RPS6KA6", "NCOA2", "CDK4", "JAK2",
    "TYK2", "GNG11", "IL10", "ANK3", "ARAF", "OSMR", "ADIPOQ", "SOCS1", "GNGT1",
    "SMPD3", "YAP1", "CDK2", "SMAD5", "FOSL1", "FZD7", "DOK2", "SYNGAP1",
    "CSNK1A1L", "IFNG", "CXCL11", "MAP3K13", "MS4A2", "PPP2R5B", "GATA2", "CDK6",
    "SAV1", "RACGAP1", "CXCL9", "ARHGAP26", "FGF7", "LPAR4", "CCND2", "NBL1",
    "STAM", "LIFR", "GNGT2", "GNA13", "CSNK2A1", "IL4R", "IL2RA", "GNB4", "IL10RB",
    "LCK", "IL27", "PTPRR", "RASGRP2", "IL20", "YWHAZ", "FZD5", "ITPR2", "ITPR1",
    "GNAZ", "NDRG1", "RGL2", "PRLR", "ATM", "CD79A", "STAT2", "WIF1", "CSNK1E",
    "PPP2R5A", "LPAR5", "AREG", "PTPN2", "NRG4", "FOXO1", "EP300", "CSNK1G1",
    "STAT4", "ELK1", "LIF", "CTF1", "TEAD1", "CALR", "SHC4", "CISH", "DGKD",
    "KSR2", "DGKZ", "GNG5", "NOTCH1", "MYD88", "KAT2B", "BMP6", "PLCG2",
    "MAPK8IP1", "HHIP", "THPO", "S1PR3", "RPS6KA3", "ADORA1", "DGKA", "TCL1A",
    "IL24", "CSF1R", "BARD1", "GNG12", "ALK", "CHEK1", "BMP5", "ACSL4", "CAMK2B",
    "ANK2", "TP63", "VEGFC", "IL12RB1", "DGKE", "TRADD", "PIK3R2", "YWHAH",
    "IL11RA", "MDM2", "STMN1", "HPSE2", "MAP3K6", "ROS1", "RASSF5", "IL13RA1",
    "CHN2", "SH3BP5", "CNTFR", "FGF19", "PKN3", "CXCL13", "CRADD", "FGF12",
    "IL12RB2", "FZD4", "FLNB", "MCL1", "CSF2RA", "IL12A", "CSF2", "CSNK2B",
    "SOCS5", "PAG1", "NR4A1", "GNG3", "PIM1", "PRKAA2", "ANK1", "EPOR", "SMO",
    "FADD", "SMARCA2", "DGKQ", "PHLPP1", "DGKH", "ZFYVE16", "RASA4", "ELK4",
    "JAK3", "IL11", "BEX1", "IFNGR1", "IL6", "LRP6", "SFRP2", "S1PR4", "IL27RA",
    "PDGFC", "RASA3", "CREB3L2", "GNG4", "PLCB4", "STK4", "FZD6", "DGKG", "WWTR1",
    "PPP2R5C", "FGF9", "GNG8", "SHH", "PLCD1", "RPS6KA1", "PRC1", "EPAS1", "MED12",
    "BTC", "CACNA1D", "CXCL10", "IL15", "GSK3A", "STAT6", "GNG7", "BTRC", "PTGS2",
    "FZD9", "NCOR1", "FGF14", "RPS6KA2", "TP53BP2", "STAT5A", "YWHAQ", "PDGFD",
    "INSR", "VLDLR", "FZD1", "S1PR5", "SFRP1", "FGF18", "PPP2R5D", "NLK", "FGF20",
    "FGFR3", "CCNE1", "RELA", "IL23A", "INHBA", "LEFTY2", "FZD8", "RPS6"
]
gene_list = [gene for gene in desired_gene_list if gene in SKCM_data.index]
for gene in desired_gene_list:
    if gene not in gene_list:
        print(f"Warning: {gene} not found in the dataset.")
SKCM_gene_data = SKCM_data.loc[gene_list]
# %%
#only use this if you want to color by something for visualizations, otherwise just use the gene data 
SKCM_metadata = metadata_df.loc[cancer_samples]
SKCM_merged = SKCM_gene_data.T.merge(
    SKCM_metadata, left_index=True, right_index=True)
# %%
#use gene data, and use merged if you want to color 
#x is the main data and y is just for coloring, not necessary
#use 2 components
X = SKCM_merged[gene_list] # gene expression only
SKCM_merged['OS_group'] = pd.qcut(
SKCM_merged['OS.time'],
q=3,
labels=["Low survival", "Medium survival", "High survival"]
)

y = SKCM_merged['OS_group']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


plt.figure(figsize=(8, 6))
sns.scatterplot(
x=X_pca[:, 0],
y=X_pca[:, 1],
hue=y,
palette="Set2",
s=80
)

plt.title("PCA of SKCM Gene Expression Data")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Survival Group")
plt.show()


# %%
