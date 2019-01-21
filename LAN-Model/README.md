# README - CFR Model - LAN
## Outline

### Folders Description
- ./data
  Contains the spec. of LANs (there are 2 versions) & cfr data
  - lan\_spc\_0.csv
  - lan\_spc\_1.csv
  - cfr data for each LANs
  - etc.
- ./ongo
  Stores the model files (.h5) during training, include some previous models
- ./release
  Stores the trained model files (.h5) & other reusable files
- \_Library & \_ProjectManifest

   import [Library](http://10.78.26.44:30000/amacs-1-1-1/Library)

### iPython Notebook & other Files Description
1. CFR\_Model\_LSTM-LAN.ipynb
   Inside this file contains the following actions:
   - Read data (cfr & spec, located in ./data & ./data-flowtool)
   - Prepare data from two normalized arrays (s\_all\_norm\_f and c\_dft\_norm)
   - Determine the feature set, version, and target training LANs
   - Train
   - Output model files (.h5) into ./ongo folder, print summary (mse, parameters, CFR curves)
   - The mse result will be put in ./log.txt

2. CFR\_Model\_LSTM-LAN\_Eva.ipynb
   This file print the evaluation information for the model trained by the first file
   - Read data (Same as 1.)
   - Prepare data (Same as 1.)
   - Determine the feature set, version, and target training LANs
   - Print nDCG & Kendall Tau, organinzed by two ways:
     - by different series (AR8, RTL, WG8, WGI)
     - by average shipping amounts

3. utils.py
   A utility script




