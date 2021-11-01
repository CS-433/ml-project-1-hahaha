# ML-Proejct1

### **Group Member**: 

Jun qing(jun.qing@epfl.ch)  Lingjun Meng(lingjun.meng@epfl.ch) Aibin Yu(aibin.yu@epfl.ch)



### **In this repository we submit:** 
**Document_code_for_pdf.ipynb**
        where all of the codes for the report's results and figures are presented. The flow is the same as the paper report.

**implementations.py:**
	where we have the basic functions as well as other functions necessary for the project

**helpers and proj1 helpers:**
	which contain auxiliar functions used in the project

**run.py:**
	File to run for the test (prediction)

**Report:** 

**Code notebook:**
	Contains all the code including: data analysis and preprocessing/cleaning, the training with the chosen method (ridge regression) and the code that is common to run.py for the predictions



### **How to run the test:**

1. Decompress two datasets in the data folder

- train.csv.zip
- test.csv.zip

```bash
unzip train.csv.zip
unzip test.csv.zip
```

2. Open run.py and change the data path in "load_data" so that it loads the desired file. 

```python
DATA_TRAIN_PATH = '../data/train.csv'
DATA_TEST_PATH = '../data/test.csv'
```

3. Open the terminal, run run.py

```bash
python run.py
```





