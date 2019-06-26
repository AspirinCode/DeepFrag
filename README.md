## DeepFrag
***
A pre-train and fine-tune method for predicting MS/MS spectra for a specific type of MS instrument.

<div align="center">
<img src="https://github.com/hcji/DeepFrag/blob/master/Img/Figure1A.png" width=400 height=270 />
</div>

### Depends:
    Anaconda for python >= 3.6
    # RDKit
    conda install -c rdkit rdkit
    # Keras
    conda install keras
    # pycdk
    pip install git+git://github.com/hcji/pycdk@master
    # pyCFMID (optional, just used for comparing with CFM-ID)
    pip install git+git://github.com/hcji/PyCFMID@master
    
### Scripts:
All scripts used for this work are included in the **Scripts** directory.    
**Scripts/train_model.py** is the scrpit which is used for pre-train the model with CFM predicted spectra. The spectra are available at [MINE](http://minedatabase.mcs.anl.gov).    
**Scripts/transfer_model_RIKEN_PlaSMA.py** is the script which is used for fine-tune the model with RIKEN PlaSMA database.    
**Scripts/example1.py** and **Scripts/example2.py** are two examples from the test dataset.   
**Scripts/annotation_example.py** and **Scripts/annotation_example2.py** are two examples explaining how to annotate the predicted spectra.   

### Graphical Interfaces
This repo doesn't contain the model. One can download the whole DeepFrag at [url](https://figshare.com/articles/DeepFrag_zip/8323568). Then use the graphical interfaces with
    cd ../DeepFrag
    python DeepFragApp.py
