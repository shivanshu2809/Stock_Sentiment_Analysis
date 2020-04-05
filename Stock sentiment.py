import pandas as pd

dataset=pd.read_csv("C:\\Users\\DELL\\Desktop\\Excelr Hyd\\Kaggle\Stock Sentiment\\Stock-Sentiment-Analysis-master\\Data.csv",encoding="ISO-8859-1")
data=dataset.iloc[:,2:27]
data.replace("[^a-zA-Z]"," ",regex=True,inplace=True)

list1=[i for i in range(25)]
new_index=[str(i) for i in list1]
data.columns=new_index

for index in new_index:
    data[index]=data[index].str.lower()
    
    
headlines=[]
for row in range(0,len(data.index)):
    headlines.append(" ".join(str(x) for x in data.iloc[row,0:25]))
    
headlines[0]


from sklearn.model_selection import train_test_split


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

countvector=CountVectorizer(ngram_range=(2,2))
traindata=countvector.fit_transform(headlines)    
y=dataset.iloc[:,1]

x_train,x_test,y_train,y_test=train_test_split(traindata,y,test_size=0.20,random_state=0)

    


model=RandomForestClassifier(n_estimators=100,criterion="entropy")
model.fit(x_train,y_train)

preds=model.predict(x_test)

from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,preds)
cm
accu=accuracy_score(y_test,preds)
accu


