# PhD-Thesis-Code

This repository contains the code and notebooks accompanying my PhD thesis:
*“Bridging Models and Mechanisms: Integrating Proteome Remodeling with Antibiotic Response”* (University of Waterloo, 2025).

The code supports:
- Simulation of bacterial growth dynamics under ribosome-targeting antibiotic pulses
- Parameter optimization using experimental delay time and RNA/Protein ratio data
- Generation of figures included in the thesis

For experimental context, see the **Materials and Methods** chapter of the thesis.

## Repository Structure
```
PhD-Thesis-Code/
├── src/                # Core Python scripts (model equations, optimization, plotting)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```
## Installation

You can set up the environment using either **Conda (recommended)** or a standard Python + pip environment.

### Option 1: Using Conda (Recommended)
```bash
conda create -n thesis-env python=3.10
conda activate thesis-env
pip install -r requirements.txt
```

### Option 2: Standard Python (pip)
Ensure Python 3.9+ is installed, then:
```bash
pip install -r requirements.txt
```

> If `requirements.txt` is not finalized yet, install the core packages manually:
> ```bash
> pip install numpy scipy matplotlib pandas jupyter
> ```

## Running the Code

You can run the code either through **Jupyter notebooks** or directly from the command line.

### With Jupyter Notebooks
1. Launch Jupyter:
```bash
jupyter notebook
```
2. Open a notebook from the `notebooks/` folder.
3. Follow the instructions and run the cells in order.

### From the Command Line
Run any of the Python scripts from the `src/` folder as needed, for example:
```bash
python src/<script_name>.py
```

Replace `<script_name>.py` with the desired script.


## Data Input

This repository **does not include lab-measured data**.

To run the optimization or simulations, users must provide their own experimental data by editing the placeholder sections inside the scripts (look for comments such as `# USER: insert your data here`).

Required formats:

- **Pulse lengths:** a Python list of integers  
  e.g. `N_values_glu = [2, 4, 6]`
- **Delay times:** a Python dictionary  
  e.g. `exp_delay_times_glu = {2: 2.5, 4: 3.4, 6: 3.8}`
- **RNA/Protein time-series (for N=4 h pulse):** a Python dictionary  
  e.g. `lab_RNA_P_glu = {4: [1.0, 1.3, 1.5, ...]}`
- **Inhibition curves:** NumPy arrays  
  e.g. `TET_concentrations = np.array([...])`  
       `growth_glucose = np.array([...])`  
       `stdev_glucose  = np.array([...])`

> Refer to the thesis **Materials and Methods** for details on how these measurements were obtained.

## Citation

If you use this repository, please cite:

- Brittany Howell, *Bridging Models and Mechanisms: Integrating Proteome Remodeling with Antibiotic Response*, University of Waterloo (2025)  
- GitHub Repository: [https://github.com/BrittanyH1997/PhD-Thesis-Code](https://github.com/BrittanyH1997/PhD-Thesis-Code)

For reproducibility, you may also cite the tagged snapshot that corresponds to the thesis submission:
- Release/Tag: `v1.0-thesis`

## License

This repository is released under the [MIT License](LICENSE).

You are free to use, modify, and distribute the code in accordance with the license terms.
Please note that no experimental data are included in this repository.
