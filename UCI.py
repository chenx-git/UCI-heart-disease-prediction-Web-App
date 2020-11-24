import os
os.chdir('/Users/chenxidong/Desktop/5001_project/')
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # drawing graphs
from sklearn.tree import DecisionTreeClassifier # a classification tree
from sklearn.tree import plot_tree # draw a classification tree
from sklearn.model_selection import cross_val_score # cross validation
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 
from sklearn.decomposition import PCA
import seaborn as sns


def main():
    st.title('🫀 Heart Diease prediction (SDSC5001 Project in 2020)')
    st.sidebar.title('🕹️ Model Tuning')
    st.markdown('Predict heart diease by three models')
    st.markdown('Data source: https://www.kaggle.com/ronitf/heart-disease-uci')
    st.sidebar.markdown('Creater: DONG,ZHANG and YU')
    #simpply use the cache last time unlesss the input changed
    @st.cache(persist=True)
    def load_data():
        data=pd.read_csv('/Users/chenxidong/Desktop/5001_project/heart.csv')
        data.columns = ['age', 
              'sex', 
              'cp', 
              'restbp', 
              'chol', 
              'fbs', 
              'restecg', 
              'thalach', 
              'exang', 
              'oldpeak', 
              'slope', 
              'ca', 
              'thal', 
              'hd']
        #one-hot encode
        data=pd.get_dummies(data,columns=['cp','restecg','slope','thal'])
        return data
    
    @st.cache(persist=True)
    #split the data to train and test
    def split(df):
        y=df.hd #target column,target vector y
        x=df.drop(columns=['hd'])
        #one-hot encoding for catagorical data
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
        return x_train,x_test,y_train,y_test
    
    #plot the evaluation metrics
    def plot_metrics(metrics_list):
        # Confusion Matrix
        if 'Confusion Matrix' in metrics_list:
            st.subheader('Confusion Matrix')
            plot_confusion_matrix(model, x_test,y_test,display_labels=class_names)
            st.pyplot()
            
        # ROC curve plot
        if 'ROC Curve' in metrics_list:
            st.subheader('ROC Curve')
            plot_roc_curve(model, x_test,y_test)
            st.pyplot()
        
        # Precision recall curve
        if 'Precision-Recall Curve' in metrics_list:
            st.subheader('Precision-Recall Curve')
            plot_precision_recall_curve(model, x_test,y_test)
            st.pyplot()
            
            
            
    df=load_data()
    x_train,x_test,y_train,y_test=split(df)
    class_names=['No heart diease','Heart diease']
    
    #explore the data before create the model
    st.sidebar.subheader("Explore the dataset")
    
    if st.sidebar.checkbox('Show Attribute Infomation',False):
        st.subheader('Attribute Info')
        from PIL import Image
        Attribute_info= Image.open('Attribute info.png')
        st.image(Attribute_info,use_column_width=True)
        
    if st.sidebar.checkbox('Display UCI dataset',False):
        st.subheader('UCI heart diease dataset')
        st.write(df)
      
    if st.sidebar.checkbox('Show Data description',False):
        st.subheader('UCI dataset description')
        st.write(df.describe())
    
    #create the model
    st.sidebar.subheader('Choose Your Model')
    classifier=st.sidebar.selectbox('Model',('Decision Tree','SVM','Logistic Regression'))
    
    #Decision Tree
    if classifier == 'Decision Tree':
        st.sidebar.subheader('Model Parameters')
        #ccp_alpha
        ccp_alpha=st.sidebar.number_input('Minimal Cost-Complexity Pruning parameter',0.001,0.080,step=0.010,key='ccp_alpha')
        max_depth=st.sidebar.number_input('Max depth of tree',5,150,step=10,key='max_depth')
        
        metrics=st.sidebar.multiselect('What metrics to plot?',('Confusion Matrix','ROC Curve','Precision-Recall Curve'))
        
        if st.sidebar.button('Run',key='classify'):
            st.subheader('Decision Tree Results')
            model=DecisionTreeClassifier(random_state=42,ccp_alpha=ccp_alpha)
            model.fit(x_train,y_train)
            accuracy=model.score(x_test, y_test)
            y_pred=model.predict(x_test)
            st.write('Accuracy: ',accuracy.round(3))
            st.write('Precision: ',precision_score(y_test, y_pred,labels=class_names).round(3))
            st.write('Recall: ',recall_score(y_test, y_pred,labels=class_names).round(3))
            plot_metrics(metrics)
            #display the optimal tree diagram
            from PIL import Image
            image = Image.open('Optimal Tree diagram.png')
            st.image(image, caption='The Optimal Tree Diagram',use_column_width=True)
           
            
            
    #SVM model        
    if classifier == 'SVM':
        st.sidebar.subheader('Model Parameters')
        C=st.sidebar.number_input('C (Regularization paramter)',0.010,10.00,step=0.010,key='C')
        kernel=st.sidebar.radio('kernel',('rbf','linear'),key="kernel")
        gamma=st.sidebar.radio("Gamma (kernel coefficient)",('scale','auto'),key='gamma')
        
        
        metrics=st.sidebar.multiselect('What metrics to plot?',('Confusion Matrix','ROC Curve','Precision-Recall Curve'))
        
        if st.sidebar.button('Run',key='classify'):
            st.subheader('SVM Results')
            model=SVC(C=C,kernel=kernel,gamma=gamma)
            model.fit(x_train,y_train)
            accuracy=model.score(x_test, y_test)
            y_pred=model.predict(x_test)
            st.write('Accuracy: ',accuracy.round(2))
            st.write('Precision: ',precision_score(y_test, y_pred,labels=class_names).round(2))
            st.write('Recall: ',recall_score(y_test, y_pred,labels=class_names).round(2))
            plot_metrics(metrics)
            
            st.subheader('PCA Visualization for real value and model predcition')
            #PCA visualization
            pca=PCA(n_components=2)
            principal_comp=pca.fit_transform(x_test)
            pca_df = pd.DataFrame(data = principal_comp, columns =['pca1','pca2'])
            y_pred= pd.DataFrame(y_pred,columns=['hd'])
            #pca_df1
            pca_df1=pca_df
            y_test=y_test.reset_index(drop=True)
            pca_df1=pd.concat([pca_df1, y_test], axis=1)
            #pca_df2
            pca_df2=pca_df
            pca_df2=pd.concat([pca_df2, y_pred], axis=1)
            plt.figure(figsize=(12,5))
            plt.subplot(1, 2, 1)
            ax = sns.scatterplot(x="pca1", y="pca2",hue = "hd", data = pca_df1, palette ='deep')
            plt.title('The real y-test value')

            plt.subplot(1, 2, 2)
            ax = sns.scatterplot(x="pca1", y="pca2",hue = "hd", data = pca_df2, palette ='deep')
            plt.title('The SVM predicted y-test value')

            plt.show()
            st.pyplot()
    
    #Logistic Regression model
    if classifier == 'Logistic Regression':
        st.sidebar.subheader('Model Parameters')
        C=st.sidebar.number_input('C (Regularization paramter)',0.000,10.00,step=0.010,key='C_LR')
        max_iter=st.sidebar.slider('Max number of iterations',100,500,key='max_iter')
        
        metrics=st.sidebar.multiselect('What metrics to plot?',('Confusion Matrix','ROC Curve','Precision-Recall Curve'))
        
        if st.sidebar.button('Run',key='classify'):
            st.subheader('Logistic Regression Results')
            model=LogisticRegression(C=C,max_iter=max_iter)
            model.fit(x_train,y_train)
            accuracy=model.score(x_test, y_test)
            y_pred=model.predict(x_test)
            st.write('Accuracy: ',accuracy.round(2))
            st.write('Precision: ',precision_score(y_test, y_pred,labels=class_names).round(2))
            st.write('Recall: ',recall_score(y_test, y_pred,labels=class_names).round(2))
            plot_metrics(metrics)
            #plot the coefficients
            st.text('The coefficients are: ')
            coef_=pd.DataFrame(model.coef_)
            coef_.columns=df.drop(columns=['hd']).columns
            st.dataframe(coef_)
            st.bar_chart(coef_)
            #display the optimal logistic regression coefficient
            from PIL import Image
            image = Image.open('Optimal_LR.png')
            st.image(image, caption='The Optimal Logistic regression coefficient',use_column_width=True)

    

if __name__ == '__main__':
    main()

st.set_option('deprecation.showPyplotGlobalUse', False)
 

