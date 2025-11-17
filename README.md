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
    Generate a table or graph which shows the comparison between non-binders and binders
    get the numbers and create a table of all figures like that paper did.
    change the variable names so looks less AI.
    is there a logical way to calculate the grey area for binding threshold 
    use a heat correltion map to produce a heatmap of the different binding affinity sources (Ki, Kd, IC50, EC50)
    For evalaution:
        could show difference between using logged binding affinity and not
        difference between using random negative sampling and not
        difference between using grey area and not
    potentially look at training on chembl as well



