pks = installed.packages()
if (!"devtools" %in% pks){
  install.packages("devtools")
}

library(devtools)
if (!"metfRag" %in% pks){
  install_github("c-ruttkies/MetFragR/metfRag")
}

library(metfRag)

generateFragments <- function(smi, mzs, neutralMonoisotopicMass, mzabs = 0.5, mzppm = 10.0, posCharge = FALSE, ionMode = 1, treeDepth = 2){
  mol <- rcdk::parse.smiles(smi)[[1]]
  frags <- metfRag::frag.generateFragments(mol, treeDepth)
  frag_smiles <- sapply(frags, rcdk::get.smiles)
  output <- unique(frag_smiles)
  return(output)
}