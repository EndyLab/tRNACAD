# tRNAShuffle
Contains code corresponding to the engineering tRNA abundances for synthetic cellular systems project. 

## Modeling code
All modeling code is available with comment in Jupyter notebooks. Colloidal dynamics and Colloidal dynamics-Computer Aided Design (CD-CAD) algorithms used throughout can be found in analysis_utils.py and are accessed within Jupyter notebooks. Files correspond to particular modeling and simulations for as follows:

ColloidalDynamics_tRNANaturalAnalysis - Figure 1 & S1: Colloidal dynamics modeling assessing wild type E. coli

ColloidalDynamics_tRNAVaryingIO - Figure 2 & S2: Colloidal dynamics modeling assessing rationally engineered tRNA distributions

CDCAD_WTecoli.npy - Figure 3: CD-CAD modeling for the E. coli transcriptome

CDCAD_RED20Ecoli.npy - Figure 4, S3, S14: CD-CAD modeling for the genome reduced E. coli transcriptome

CDCAD_RED20_SingleGeneOptimization - Figure 5, S14: Optimization and analysis of tRNA distributions for RED20-encoded GFP

CDCAD_WT_PerturbationAnalysis.npy - Figure S9, S10: CD-CAD perturubation analysis for WT E. coli

CDCAD_ompF_PerturbationAnalysis.npy - Figure S12: CD-CAD perturbation analysis for the highest expressing gene in the E. coli transcriptome, ompF

CDCAD_RED20GFP_PerturbationAnalysis - Figure S13: CD-CAD perturubation analysis for RED20-encoded GFP

CDCAD_genestratification - Figure S11: Gene stratficiation analysis

## Experimental analysis code
Code for experimental analysis (Figure 5 and related supplemental figures) can be found in experimental_data_analysis.ipynb

