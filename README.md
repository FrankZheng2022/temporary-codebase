
# temporary-codebase LIBERO

### Instructions for installing libero

**Step 1: Setup Environment**: 
```
conda create -n libero python=3.8.13
conda activate libero
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e .
```


**Step 2: Download Libero Dataset**: 
```
cd LIBERO
python benchmark_scripts/download_libero_datasets.py --datasets libero_100
```

**Step 3: Convert Dataset**: 
Copy the ``utils/convert_data.py`` file into the LIBERO repo, and then run
```
python convert_data.py --save_path DATASET_PATH --libero_path ${LIBERO_PATH}/LIBERO/libero/datasets/libero_90
python convert_data.py --save_path DATASET_PATH --libero_path ${LIBERO_PATH}/LIBERO/libero/datasets/libero_10
```
Here DATASET_PATH is the path of the libero dataset that you are going to store.


**Step 4: Install the additional dependencies of our codebase**:
``
pip install -r reuiqrements_210.txt
``

