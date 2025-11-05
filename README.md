# DTI-Predictor
This is my final year project where I use machine learning to predict the binding success rate between drug molecules and protein targets. I will code the frontend off of the react js framework and use a REST API to connect it to the backend - Neural network, Database

For Data:

# 1. Install dependencies (if not already done)
cd Backend
pip install -r requirements.txt

# 2. Run the data preparation script
python -m app.data.download_and_prepare

https://www.kaggle.com/datasets/christang0002/davis-and-kiba/data

for protein sequences:
	1.	Pull the UniProt ID from BindingDB.
	2.	Query UniProt for the amino acid sequence.
	3.	Combine the two into your dataset.

for drug SMILES:
    1.  rest api json output

Use ProtBERT for Feature Extraction of protein sequences

Use RdKIT to convert SMILES to MACCS keys


Next Steps:
    change the variable names so looks less AI.
    need to calculate the grey area for binding threshold 
    potentially look at training on chembl as well
    use Protbert for feature extraction
    use RDKit for SMILES to MACCS keys
