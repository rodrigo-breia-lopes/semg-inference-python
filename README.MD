# sEMG inference with python

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.11.6 installed on your system.

## Installation

To install the repo and required packages, follow the following steps:

Clone the repository:

``` bash
git clone https://github.com/rodrigo-breia-lopes/semg-inference-python.git
```

Navigate to the repo folder:

``` bash
cd semg-inference-python
```

Create and activate the virtual environment:

```bash
python -m venv env
source env/bin/activate
```

Install required packages: 

```bash
pip install -r requirements.txt
```
## Download the dataset

Do the next steps in order to download the Ninapro dataset, from the [official website](https://ninapro.hevs.ch/). The shell script `batch_down.sh` automatically downloads each subject's data, and puts in the appropriate folder, to avoid manual download.

First enable the execution of the script:
```bash
sudo chmod +x batch_down.sh
```

Run the script (inside the repo)
```bash
./batch_down.sh
```
After the script is done, the dataset can be found in the `datasets` folder.
 
 ## Usage
 
Add the environment to ipykernel
```bash
python -m ipykernel install --user --name=env --display-name="Python (semg-inference)"
```

Open jupyter
```bash
jupyter lab
```
