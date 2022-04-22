"""
Example Usage:
python phish-bert-20211006-script.py
"""
import cudf;
from sklearn.model_selection import train_test_split;
from binary_sequence_classifier import BinarySequenceClassifier;
import s3fs;
from os import path;

def preprocessing():
    CLAIR_TSV = "Phishing_Dataset_Clair_Collection.tsv"
    SPAM_TSV = "spam_assassin_spam_200_20021010.tsv"
    EASY_HAM_TSV = "spam_assassin_easyham_200_20021010.tsv"
    HARD_HAM_TSV = "spam_assassin_hardham_200_20021010.tsv"
    ENRON_TSV = "enron_10000.tsv"
    S3_BASE_PATH = "rapidsai-data/cyber/clx"

    print("Preprocessing...")
    # Clair dataset
    if not path.exists(CLAIR_TSV):
        fs = s3fs.S3FileSystem(anon=True)
        fs.get(S3_BASE_PATH + "/" + CLAIR_TSV, CLAIR_TSV)
    
    dfclair = cudf.read_csv(CLAIR_TSV, delimiter='\t', header=None, names=['label', 'email']).dropna()

    # Phishing emails of the SPAM ASSASSIN dataset
    if not path.exists(SPAM_TSV):
        fs = s3fs.S3FileSystem(anon=True)
        fs.get(S3_BASE_PATH + "/" + SPAM_TSV, SPAM_TSV)
 
    dfspam = cudf.read_csv(SPAM_TSV, delimiter='\t', header=None, names=['label', 'email'])
    
    # Benign emails of the SPAM ASSASSIN dataset
    if not path.exists(EASY_HAM_TSV):
        fs = s3fs.S3FileSystem(anon=True)
        fs.get(S3_BASE_PATH + "/" + EASY_HAM_TSV, EASY_HAM_TSV)

    dfeasyham = cudf.read_csv(EASY_HAM_TSV, delimiter='\t', header=None, names=['label', 'email'])
    # Benign emails of the SPAM ASSASSIN dataset that are easy to be confused with phishing emails
    if not path.exists(HARD_HAM_TSV):
        fs = s3fs.S3FileSystem(anon=True)
        fs.get(S3_BASE_PATH + "/" + HARD_HAM_TSV, HARD_HAM_TSV)

    dfhardham = cudf.read_csv(HARD_HAM_TSV, delimiter='\t', header=None, names=['label', 'email'])
    # Benign Enron emails

    if not path.exists(ENRON_TSV):
        fs = s3fs.S3FileSystem(anon=True)
        fs.get(S3_BASE_PATH + "/" + ENRON_TSV, ENRON_TSV)

    dfenron = cudf.read_csv(ENRON_TSV, delimiter='\t', header=None, names=['label', 'email'])
    
    df_total = cudf.concat([dfhardham, dfeasyham, dfspam, dfclair, dfenron], ignore_index=True)

    X_train, X_test, y_train, y_test = train_test_split(df_total, df_total['label'], train_size=0.8)
    return(X_train, y_train, X_test, y_test)



def main():

    
    X_train, y_train, X_test, y_test=preprocessing()
    

    print("Model Loading...")
    seq_classifier = BinarySequenceClassifier()
    seq_classifier.init_model("bert-base-uncased")

    print("Model Training...")
    seq_classifier.train_model(X_train["email"], y_train, epochs=3)

    print("Saving Model")
    seq_classifier.save_model("./phish-bert-20211006")
    
    print("Model Evaluation")
    print("Accuracy:")
    print(seq_classifier.evaluate_model(X_test["email"], y_test))


main()

