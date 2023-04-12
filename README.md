# To make summary data (this is only necessary after a new experiment, and it will take 30 minutes):
* Make sure you have raw_data/master_file.pkl
* Run python make_summary_data.py
# This creates the following files in ./summary_data:
* targets.csv

    This file has columns "m","targetID","used","0","1",..."29"
    where "m" is the network size, "targetID" is the name of the m-dependent index of the target function, "used" is True if the target was used in the experiment, and the other columns are the 30 values of the target.

* summary.csv

    This file has columns "m", "targetID", "KID", "dimerID", "Linf", "MSE", "goodenough", 
    where "m" is the network size, "targetID" is the name of the m-dependent index of the target function, "KID" is the name of the m-dependent index of the network parameter K, "dimerID" is the name of the index of the dimer, "Linf" is the Linf error of the target fit, "MSE" is the MSE error of the target fit, and "goodenough" is True if Linf <= 1.0.

* a_opt.pkl (This file is too large to store on Github...run `python make_summary_data.py` locally to generate or download it from [here](https://www.dropbox.com/s/lpk8bgbzodn8xoh/a_opt.pkl?dl=0)

    This file is a dictionary with keys "m" and a value of a 4D numpy array of shape (N K's x N Dimers x N targets x N accessories). The first index is the index of the KID, the second index is the index of the dimer, the third index is the index of the target, and the fourth index is the index of the accessory. The value is the optimal accessory parameter a as determined by the optimization experiment.

* K_random.pkl

    This file is a dictionary with keys "m" and a value of a 2D numpy array of shape (N K's x N Dimers). The first index is the index of the KID, and the second index is the index of the dimer. The value is the random K value used in the experiment.

* name_dict.pkl

    This file is a dictionary with keys "K_names" and "a_names" and values of dictionaries with keys "m" and values of lists of names. The "K_names" dictionary has keys "m" and values of lists of names of the KID's. The "a_names" dictionary has keys "m" and values of lists of names of the dimerID's.

