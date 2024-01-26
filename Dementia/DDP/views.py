from django.shortcuts import render

def index(request):
    return render(request,'index.html')

import pandas as pd
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import  LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.metrics import accuracy_score
import numpy as np


def dementia(request):
    if(request.method=="POST"):
        data=request.POST
        algo=data.get('alg')
        mrid=data.get('MRI ID')
        gndr=data.get('Gender')
        age=data.get('Age')
        edu=data.get('EDUC')
        ses=data.get('SES')
        mmse=data.get('MMSE')
        cdr=data.get('CDR')
        etiv=data.get('eTIV')
        nwbv=data.get('nBWV')
        
        if algo == '0':  # Check for '0', which corresponds to BRandom
            result, acc = BRandom(mrid,gndr,age,edu,ses,mmse,cdr,etiv,nwbv)
        elif algo == '1':  # Check for '1', which corresponds to BNaive
            result, acc = BNaive(mrid,gndr,age,edu,ses,mmse,cdr,etiv,nwbv)
        else:
            result, acc = BDTree(mrid,gndr,age,edu,ses,mmse,cdr,etiv,nwbv)
        return render(request, "dementia.html", context={'result': result, 'acc': acc})
    return render(request,'dementia.html')


def BRandom(mrid,gndr,age,edu,ses,mmse,cdr,etiv,nwbv):
    path="C:/Users/KAVYA/Desktop/Demantia_project/21_dementiadiseasetypeprediction/dementia_dataset.csv"
    df=pd.read_csv(path)

    df1=df.loc[df['Group']=='Converted']
    df=df.drop(df1.index)
    df1.head()

    df1['Last_Visit']=df1.groupby('Subject ID')['Visit'].transform('max')
    df1.loc[df1['Visit'] < df1['Last_Visit'],'Group']='Nondemented'
    df1.loc[df1['Visit']==df1['Last_Visit'],'Group']='Demented'
    df1.drop('Last_Visit',axis=1,inplace=True)
    df1.head()

    frames=[df,df1]
    df=pd.concat(frames)
    df['Group'].unique()

    df.rename(columns={'M/F':'Gender'},inplace=True)
    df.drop(columns=['Subject ID','Hand','Visit','MR Delay'],inplace=True)
    df.SES.fillna(df.SES.mode()[0],inplace=True)
    df.MMSE.fillna(df.MMSE.mean(),inplace=True)
    df.drop(columns=['ASF'],inplace=True)

    le=LabelEncoder()
    df['Gender']=le.fit_transform(df['Gender'])
    df['Group']=le.fit_transform(df['Group'])
    df['MRI ID']=le.fit_transform(df['MRI ID'])

    inputs=df.drop(['Group'],axis=1)
    output=df.Group
    X_train,X_test,y_train,y_test=train_test_split(inputs,output,test_size=0.2,random_state=42)

    #Random Forest

    model=RandomForestClassifier(random_state=42)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    result=model.predict([[mrid,gndr,age,edu,ses,mmse,cdr,etiv,nwbv]])
    acc = accuracy_score(y_test, y_pred)
    acc=result[0],acc*100
    if (result==0):
        result="Person Is Demented"
    else:
        result="Person Is NonDemented"

    return result, acc

def BNaive(mrid,gndr,age,edu,ses,mmse,cdr,etiv,nwbv):
    path="C:/Users/KAVYA/Desktop/Demantia_project/21_dementiadiseasetypeprediction/dementia_dataset.csv"
    df=pd.read_csv(path)

    df1=df.loc[df['Group']=='Converted']
    df=df.drop(df1.index)
    df1.head()

    df1['Last_Visit']=df1.groupby('Subject ID')['Visit'].transform('max')
    df1.loc[df1['Visit'] < df1['Last_Visit'],'Group']='Nondemented'
    df1.loc[df1['Visit']==df1['Last_Visit'],'Group']='Demented'
    df1.drop('Last_Visit',axis=1,inplace=True)
    df1.head()

    frames=[df,df1]
    df=pd.concat(frames)
    df['Group'].unique()

    df.rename(columns={'M/F':'Gender'},inplace=True)
    df.drop(columns=['Subject ID','Hand','Visit','MR Delay'],inplace=True)
    df.SES.fillna(df.SES.mode()[0],inplace=True)
    df.MMSE.fillna(df.MMSE.mean(),inplace=True)
    df.drop(columns=['ASF'],inplace=True)

    le=LabelEncoder()
    df['Gender']=le.fit_transform(df['Gender'])
    df['Group']=le.fit_transform(df['Group'])
    df['MRI ID']=le.fit_transform(df['MRI ID'])

    inputs=df.drop(['Group'],axis=1)
    output=df.Group
    X_train,X_test,y_train,y_test=train_test_split(inputs,output,test_size=0.2,random_state=42)

    #Random Forest

    nb=GaussianNB()
    nb.fit(X_train,y_train)
    y_pred=nb.predict(X_test)
    result=nb.predict([[mrid,gndr,age,edu,ses,mmse,cdr,etiv,nwbv]])
    acc = accuracy_score(y_test, y_pred)
    acc=result[0],acc*100
    if (result==0):
        result="Person Is Demented"
    else:
        result="Person Is NonDemented"

    return result, acc


def BDTree(mrid,gndr,age,edu,ses,mmse,cdr,etiv,nwbv):
    path="C:/Users/KAVYA/Desktop/Demantia_project/21_dementiadiseasetypeprediction/dementia_dataset.csv"
    df=pd.read_csv(path)

    df1=df.loc[df['Group']=='Converted']
    df=df.drop(df1.index)
    df1.head()

    df1['Last_Visit']=df1.groupby('Subject ID')['Visit'].transform('max')
    df1.loc[df1['Visit'] < df1['Last_Visit'],'Group']='Nondemented'
    df1.loc[df1['Visit']==df1['Last_Visit'],'Group']='Demented'
    df1.drop('Last_Visit',axis=1,inplace=True)
    df1.head()

    frames=[df,df1]
    df=pd.concat(frames)
    df['Group'].unique()

    df.rename(columns={'M/F':'Gender'},inplace=True)
    df.drop(columns=['Subject ID','Hand','Visit','MR Delay'],inplace=True)
    df.SES.fillna(df.SES.mode()[0],inplace=True)
    df.MMSE.fillna(df.MMSE.mean(),inplace=True)
    df.drop(columns=['ASF'],inplace=True)

    le=LabelEncoder()
    df['Gender']=le.fit_transform(df['Gender'])
    df['Group']=le.fit_transform(df['Group'])
    df['MRI ID']=le.fit_transform(df['MRI ID'])

    inputs=df.drop(['Group'],axis=1)
    output=df.Group
    X_train,X_test,y_train,y_test=train_test_split(inputs,output,test_size=0.2,random_state=42)

    #Random Forest

    dt=tree.DecisionTreeClassifier()
    dt.fit(X_train,y_train)
    y_pred=dt.predict(X_test)
    result=dt.predict([[mrid,gndr,age,edu,ses,mmse,cdr,etiv,nwbv]])
    acc = accuracy_score(y_test, y_pred)
    acc=result[0],acc*100
    if (result==0):
        result="Person Is Demented"
    else:
        result="Person Is NonDemented"

    return result, acc


# Create your views here.