Emo-FilM 
This repository contains the code used for processing of the Emo-FilM dataset

Scripts for bidsification:
- BIDSification/Heurist_phys.py
- BIDSification/fixtimeunits.py
- BIDSification/steps.py

Calculation of consensus annotation and quality control of annotations:
- MakeMasterfile.py : this reads the annotation files as they are in the published dataset and calculated the consensus annotation. In doing so it also computes all the quality control metrics.
- AnnotationQC_plots.py : this used the outputs from MakeMasterfile to plot the quality control metrics for the annotations.

Validation of film annotations:
- AnnotationVal.py : this reads the validation files and the consensus annotations they are in the published dataset and computes and plots quality control metrics for the validation

fMRI preprocessing:
- Statistical analysis/PP_design.fsf : this is a template file to be used in FEAT (fsl) for the preprocessing of our files. Replace the relevant filepaths (e.g., using bash)
  
Two helper scripts containing functions and constants:
- helper_annot.py : contains two functions used in MakeMasterfile.py and AnnotationsVal.py
- constants_emo_film.py : contains a number of constants used in MakeMasterfile.py, AnnotationQC_plots.py and AnnotationsVal.py, such as the films and item names

Stimulus presentation:
- Scripts/*
- ValidationScript/*
