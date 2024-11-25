
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
parkinson_data=pd.read_csv('D:\sindhuja\parkinsons.data')
parkinson_data=parkinson_data.drop(columns=['name'],axis=1)
x=parkinson_data.drop(columns=['status'],axis=1)
y=parkinson_data['status']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)

print(x.shape,x_train.shape,x_test.shape)

scaler=StandardScaler()
scaler.fit(x_train)

x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

print(x_train)

model=svm.SVC(kernel='linear')

#training the svm model with training data
model.fit(x_train,y_train)



## accuracy score on training data
x_train_prediction=model.predict(x_train)
training_data_accuracy=accuracy_score(y_train,x_train_prediction)

print("Accuracy score of training data: ",training_data_accuracy)

## accuracy score on test data
x_test_prediction=model.predict(x_test)
test_data_accuracy=accuracy_score(y_test,x_test_prediction)

print("Accuracy score of test data: ",test_data_accuracy)

input_data=(202.26600,211.60400,197.07900,0.00180,0.000009,0.00093,0.00107,0.00278,0.00954,0.08500,0.00469,0.00606,0.00719,0.01407,0.00072,32.68400,0.368535,0.742133,-7.695734,0.178540,1.544609,0.056141)


## changing input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

## reshape the numpy array
input_data_reshaped= input_data_as_numpy_array.reshape(1,-1)

## standarize the data
std_data= scaler.transform(input_data_reshaped)

prediction= model.predict(std_data)
print(prediction)

if(prediction[0]==0):
    print("person is not having parkinsons disease")
else:
    print("person is having parkinsons disease")

st.markdown("<h1 style='text-align: center; font-size:35px;'>PARKINSON'S DISEASE PREDICTION</h1>",
 unsafe_allow_html=True,)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.text("            by Sindhuja")
col1, col2 = st.columns(2)
with col1:
 a = st.number_input("MDVP:Fo(Hz)")
 b = st.number_input("MDVP:Fhi(Hz)")
 c = st.number_input("MDVP:Flo(Hz)")
 d = st.number_input("MDVP:Jitter(%)")
 e = st.number_input("MDVP:Jitter(Abs)")
 f = st.number_input("MDVP:RAP")
 g = st.number_input("MDVP:PPQ")
 h = st.number_input("Jitter:DDP")
 i = st.number_input("MDVP:Shimmer")
 j = st.number_input("MDVP:Shimmer(dB)")
k = st.number_input("Shimmer:APQ3")
with col2:
 l = st.number_input("Shimmer:APQ5")
 m = st.number_input("MDVP:APQ")
 n = st.number_input("Shimmer:DDA")
 o = st.number_input("NHR")
 p = st.number_input("HNR")
 r = st.number_input("RPDE")
 s = st.number_input("DFA")
 t = st.number_input("spread1")
 u = st.number_input("spread2")
 v= st.number_input("D2")
w = st.number_input("PPE")
if st.button("Predict Parkinson's Disease"):
    input_data = (a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,r,s,t,u,v,w)
    input_data_as_numpy_array= np.asarray(input_data)
    input_data_reshaped= input_data_as_numpy_array.reshape(1,-1)
    std_data= scaler.transform(input_data_reshaped)
    prediction= model.predict(std_data)
    print(prediction)
    if(prediction==0):
      aa='person is not having parkinsons disease'
      st.success(aa)
    else:
      bb="person is having parkinsons disease"
      st.success(bb)
