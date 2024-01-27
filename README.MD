# sEMG inference with python

## Setup

To use the repo, do the following:

- Clone the repo
```
git clone https://github.com/rodrigo-breia-lopes/semg-inference-python.git
```

- Go to the repo directory
```
cd semg-inference-python
```
- Install python 3.11.6
- Create a virtual environment
```bash
python -m venv env
```
- Activate the virtual environment
```bash
source env/bin/activate
```
- Install required packages
```bash
pip install -r requirements.txt
```
- Download the dataset
First enable the execution of the script:
```bash
sudo chmod +x batch_down.sh
```

- Run the script (inside the repo)
```bash
./batch_down.sh
```

- Use jupyter-lab

Add the environment to ipykernel
```bash
python -m ipykernel install --user --name=env --display-name="Python (semg-inference)"
```

Open jupyter
```bash
jupyter lab
```
