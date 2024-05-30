 # Online Learning and Information Exponents: The Importance of Batch size & Time/Complexity Tradeoffs
<div width=auto>
    <img src="figures/SQ-staircase.png" width=100%>
</div>

> Example of the new staircase structure emerging in algorithms that repeats data. $h^\star_{sign}=\mathrm{sign}(x_1x_2x_3)$ cannot be learned in $O(d)$ steps, while $h^\star_{stair}=h^\star_{sign}+\mathrm{He}_4(x_1)$ can because of the staircaise mechanism.


### Installation
Tested with Python 3.11
```
git submodule update --init --recursive # install boostmath
pip install -r requirements.txt
pip install -e giant-learning --no-binary :all:
```

### How to use
The file structure of this repository is as follows:
 - `giant-learning/` contains the Python package used to run the experiments.
 - `hyperparameters/` contains some example configuration files needed to run the experiments.
 - `running.py` is the main script to run the experiments.
 - `plotting.py` is the main script to plot the results.
 - `example.ipynb` is a Jupyter notebook that shows how to run the experiments and plot the results.
