"""
Example Usage:
python hammah-inference.py --validationdata hammah-user123-validation-data.csv --model hammah-user123-20211017-dill.pkl --output abp-validation-output.csv
"""

import pandas as pd
import os
import numpy as np
import glob
import torch
from dfencoder import AutoEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import dill
import argparse
import json
#from pandas import json_normalize
from pandas.io.json import json_normalize
import cudf
import datetime
import cupy as cp
from clx.analytics.stats import rzscore
import clx.analytics.periodicity_detection as pdd





def infer(validationdata,model,output):
    
    def zscore(data):
        mu = cp.mean(data)
        std = cp.std(data)
        return (data - mu) / std

    def date2min(time):
        start = START
        timesince = time - start
        return int(timesince.total_seconds() // 60)

    form = "%Y-%m-%dT%H:%M:%SZ"
    def stript(s):
        obj = datetime.datetime.strptime(s, form)
        return obj

    def back_to_string(obj):
        return "{}-{}:{}:{}".format(f"{obj.month:02}", f"{obj.day:02}", f"{obj.hour:02}", f"{obj.minute:02}")
    
    
    
    
    
    
    
    

    X_val=pd.read_csv(validationdata)


    col_list=['userIdentityaccountId','eventSource','eventName','sourceIPAddress','userAgent','userIdentitytype','apiVersion', 'userIdentityprincipalId', 'userIdentityarn', 'userIdentityaccessKeyId', 'tlsDetailsclientProvidedHostHeader', 'errorCode','errorMessage','userIdentitysessionContextsessionIssueruserName']

    for i in list(X_val):
        if i not in col_list:
            X_val=X_val.drop(i,axis=1)

    with open(model, 'rb') as f:
         model = dill.load(f)

    scores = model.get_anomaly_score(X_val)
    X_val['ae_anomaly_score'] = scores

    X_val['ae_anomaly_score'] = scores
    X_val.sort_values('ae_anomaly_score', ascending=False).head(10)
    #since inference is done, add the original columns back so the output will be the same as the input format
    #X_val['ts_anomaly']=X_val_original['ts_anomaly']
    df = cudf.read_csv("hammah-user123-validation-data.csv")
    df = df.sort_values(by=['eventTime'])
    timearr = df.eventTime.to_array()
    START = stript(timearr[0])
    END = stript(timearr[-1])
    timeobj = list(map(stript, timearr))
    hs = list(map(date2min, timeobj))
    n, bins = cp.histogram(cp.array(hs), bins = cp.arange(0, max(hs)))
    signal = cudf.Series(n)
    a = cp.fromDlpack(signal.to_dlpack())
    periodogram = pdd.to_periodogram(signal)
    periodogram = periodogram[:int((len(signal)/2))]
    threshold = float(cp.percentile(cp.array(periodogram), 90))
    indices = cudf.Series(cp.arange(len(periodogram)))[periodogram < threshold].to_array()
    rft= cp.fft.rfft(a)
    rft[indices] = 0
    recon = cp.fft.irfft(rft)
    err=(abs(recon-a))
    z = zscore(err)
    indices = cudf.Series(cp.arange(len(z)))[z >= 8].to_array()
    strlist=[]
    for mins in indices:
        from_start = START + datetime.timedelta(minutes=int(mins))
        strlist.append(back_to_string(from_start))
    df['ts_anomaly'] = False
    for i in strlist:
        df['ts_anomaly'] = df['eventTime'].str.contains(i)
    X_val_original=X_val
    X_val.insert(0, 'eventID',X_val.index)
    X_val.insert(0, '',X_val.index)
    X_val['ts_anomaly']=df['ts_anomaly'].to_pandas()
    X_val.to_csv(output,index=False)


def main():

    infer(args.validationdata,args.model,args.output)
    

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validationdata", required=True,help="Labelled data in JSON format")
    parser.add_argument("--model", required=True, help="trained model")
    parser.add_argument("--output", required=True, help="output filename")
    args = parser.parse_args()

main()
