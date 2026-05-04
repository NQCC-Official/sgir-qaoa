# SGIR-QAOA: Spectral Gap Informed Ramp QAOA
[![arXiv](https://img.shields.io/badge/arXiv-2604.24580-b31b1b.svg)](https://scirate.com/arxiv/2604.24580)

**Spectral Gap Informed Ramp QAOA (SGIR-QAOA)** builds upon Linear Ramp QAOA (LR-QAOA) where we use spectral gap information from an adiabatic Hamiltonian, with the QAOA mixer Hamiltonian as our initial Hamiltonian, to make smooth ramps.


---

## 📂 Project Structure

```text
.
├── 📁 paper_code/          # Source code to reproduce paper results
├── 📁 SGIR_tutorial/       # Folder for tutorial
│   └── SGIR_tutorial.ipynb # Interactive notebook tutorial
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
```

---

## 🚀 Getting Started

### 🛠️ Installation
Before running the code or the tutorial, ensure you have the necessary dependencies installed. 

1. Clone the repository
First, download the project files to your local machine

2. Set up a virtual environment
It is recommended to use a virtual environment to keep your dependencies organized:

```bash
# Create the virtual environment
python -m venv sgir-qaoa_env

# Activate the environment
# On Windows:
.\sgir-qaoa_env\Scripts\activate
# On macOS/Linux:
source sgir-qaoa_env/bin/activate
```

3. Install required packages

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```


### Interactive Tutorial
If you are new to SGIR-QAOA, we recommend starting with the interactive tutorial. Navigate to the [`SGIR_tutorial/`](./SGIR_tutorial/) folder and open `SGIR_tutorial.ipynb` to see a step-by-step demonstration of the SGIR-QAOA implementation.

### Reproducing Results
The scripts used to generate the data and figures presented in our paper are located in the [`paper_code/`](./paper_code/) directory.

---

## 📖 Citation
If you find this work useful for your research, please cite our paper:
```bibtex
@article{sgir_qaoa_2026,
  title={Spectral Gap Informed Ramp QAOA},
  author={Kieran McDowall and Konstantinos Georgopoulos and Petros Wallden},
  journal={arXiv preprint arXiv:2604.24580},
  year={2026},
  url={[https://arxiv.org/abs/2604.24580](https://arxiv.org/abs/2604.24580)}
}
