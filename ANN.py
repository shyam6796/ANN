import sys 
import pandas as pd
import numpy as np
import warnings
from sklearn import datasets
import pylab as pl
from sklearn.exceptions import DataConversionWarning
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors, linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap  
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

def generate_accuracy(model, x_train,x_test,y_train,y_test,k):
	select_feature = SelectKBest(chi2, k).fit(x_train, y_train)
	selected_features_data = pd.DataFrame({'Feature':list(x_train.columns),'Scores':select_feature.scores_})
	x_train_chi = select_feature.transform(x_train)
	x_test_chi = select_feature.transform(x_test)
	lr_chi_model = model.fit(x_train_chi,y_train.values.ravel())
	ac = accuracy_score(y_test,lr_chi_model.predict(x_test_chi))
	f_score = f1_score(y_test,lr_chi_model.predict(x_test_chi))
	print('------number of features: {}------ \n'.format(k))
	print('Accuracy is: ', ac)
	print('F1 score is: ', f_score)
	print('--------------------------------- \n')
	return 1

	
	
#main:
def main():
	names = ['Sample ID','Clump thickness','Uniformity of cell size','Uniformity of cell shape','Marginal adhesion','Single epithelial cell size','Number of bare nuclei','Bland chromatin','Number of normal nuclei','Mitosis','diagnosis']
	data = pd.read_csv('breast-cancer-wisconsin.csv',names =names).drop_duplicates(keep='first')
	index = data[ data['Number of bare nuclei'] == '?'].index
	data.drop(index, inplace=True)
	data.drop('Sample ID', axis=1, inplace=True)
	data['Number of bare nuclei'] = pd.to_numeric(data['Number of bare nuclei'], errors='coerce')
	data['diagnosis'][data['diagnosis'] == 2] = 0
	data['diagnosis'][data['diagnosis'] == 4] = 1
	print('\n---------------------------- dataset info ----------------------------- \n')
	print(data.info())
	print('-----------------------------------------------------------------------\n\n\n')
	
	#graph heat map
	"""sns.set(style='ticks', color_codes=True)
	plt.figure(figsize=(12, 10))
	sns.heatmap(data.astype(float).corr(), linewidths=0.1, square=True, linecolor='white', annot=True)
	plt.show()"""
	
	#graph bar charts 
	"""fig = plt.figure()
	ax = sns.countplot(x='Clump thickness', hue='diagnosis', data=data)
	ax.set(xlabel='Clump thickness', ylabel='No of cases')
	fig.suptitle("Clump thickness w.r.t. Class", y=0.96)
	plt.show()"""
	
	#density plot
	"""
	p1=sns.kdeplot(data.loc[(data['diagnosis']== 0),'Clump thickness'], color='g', shade=True, Label='benign') 
	p1=sns.kdeplot(data.loc[(data['diagnosis']== 1),'Clump thickness'], color='r', shade=True, Label='maligan')
	p1.set(xlabel='Clump thickness', ylabel='diagnosis')
	plt.show() """
	
	
	
	print('\n--------------------------target value info --------------------------- \n')
	print('0 = benign & 1 = malignan ')
	print(data.diagnosis.value_counts())
	print('-----------------------------------------------------------------------\n\n\n')
	
	#Univariate Selection 
	
	X = data.loc[:, data.columns != 'diagnosis']
	Y = data.loc[:, data.columns == 'diagnosis']
	x_train, x_test, y_train, y_test = train_test_split(X, Y,test_size=0.3,random_state=8)
	clf_lr = LogisticRegression(solver='liblinear')
	lr_baseline_model = clf_lr.fit(x_train,y_train.values.ravel())
	ac = accuracy_score(y_test,lr_baseline_model.predict(x_test))
	f_score = f1_score(y_test,lr_baseline_model.predict(x_test))
	print('\n----------------------basic logestic Regression------------------------ \n')
	print('Accuracy is: ', ac)
	print('F1 score is: ', f_score)
	print('-----------------------------------------------------------------------\n\n\n')
	
	select_feature = SelectKBest(chi2, 5).fit(x_train, y_train)
	selected_features_data = pd.DataFrame({'Feature':list(x_train.columns),'Scores':select_feature.scores_})
	print('\n----------------------------feature scores----------------------------- \n')
	print(selected_features_data.sort_values(by='Scores', ascending=False))
	print('-----------------------------------------------------------------------\n\n\n')
	
	for k in range(1,9):
		generate_accuracy(clf_lr,x_train, x_test,y_train, y_test,k)
	
	#Recursive feature elimination with cross validation
	rfe = RFE(estimator=clf_lr, step=1)
	rfe = rfe.fit(x_train, y_train.values.ravel())
	selected_rfe_features = pd.DataFrame({'Feature':list(x_train.columns),'Ranking':rfe.ranking_})
	selected_rfe_features.sort_values(by='Ranking')
	x_train_rfe = rfe.transform(x_train)
	x_test_rfe = rfe.transform(x_test)
	lr_rfe_model = clf_lr.fit(x_train_rfe, y_train.values.ravel())
	rfecv = RFECV(estimator=clf_lr, step=1, cv=6, scoring='accuracy')
	rfecv = rfecv.fit(x_train, y_train.values.ravel())
	print('\n-----------optimal feature scores based on cross validation----------- \n')
	print('Optimal number of features :', rfecv.n_features_)
	print('\n')
	print('Best features :', x_train.columns[rfecv.support_])
	print('-----------------------------------------------------------------------\n\n\n')
	
	"""plt.figure()
	plt.xlabel("Number of features selected")
	plt.ylabel("Cross validation score of number of selected features")
	plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
	plt.show()"""
	

	
	#PCA
	"""
	breast_cancer_data = datasets.load_breast_cancer()
	#breast_cancer_data
	x = breast_cancer_data.data
	y = breast_cancer_data.target
	pca = PCA(n_components=2)
	pca2 = pca.fit_transform(x)
	#2DPlot for PCA1 and PCA2 
	plt.figure(figsize=(15,10))
	plt.scatter(pca2[:,0], pca2[:,1],c=y, edgecolor='none', alpha=0.7,
			cmap=plt.get_cmap('jet', 10), s=20)
	plt.colorbar()
	plt.xlabel('principal component 1')
	plt.ylabel('principal component 2')
	plt.savefig('2D Plot_PCA1_PCA2.png')
	plt.show()
	plt.close()"""
	
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
			 finalDf = pd.concat([principalDf, df[['target']]], axis = 1)
			 fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['2','4']
colors = ['g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
	
	
	
	

	#knn for  all feature of dataset
	model = KNeighborsClassifier(n_neighbors=7)
	model.fit(x_train, y_train)
	#predict & score
	y_predict = model.predict(x_test)
	score=model.score(x_test, y_test)      
    #– Cross validation
	model_cv = KNeighborsClassifier(n_neighbors=7)
	cv_scores = cross_val_score(model_cv, X, Y, cv=5)
        
	"""k_range = range(1, 30)
	k_scores = []
	for k in k_range:
		knn = KNeighborsClassifier(n_neighbors=k)
		scores = cross_val_score(knn, X, Y, cv=5, scoring='accuracy')
		k_scores.append(scores.mean())
	plt.plot(k_range, k_scores)
	plt.xlabel('Value of K for KNN')
	plt.ylabel('Cross-Validated Accuracy')
	plt.show()"""
	#grid search
	model2 = KNeighborsClassifier()
	param_grid = {'n_neighbors': np.arange(1, 30)}
	knn_gscv = GridSearchCV(model2, param_grid, cv=5)
	knn_gscv.fit(X, Y)
	#all values 
	print('\n-------------------------KNN for all features-----------------------\n')
	print('----------------------score for n=7 ----------------------\n')
	print('accuracy: {}'.format(score))
	print('----------------------------------------------------------\n')
	print('------------cross validationscore for n=7 cv=5------------\n')
	print(cv_scores)
	print('\n')
	print('cv_scores mean: {}'.format(np.mean(cv_scores)))
	print('----------------------------------------------------------\n')
	print('--------------------optimal n for KNN---------------------\n')
	print('optimize n for knn:  {}'.format(knn_gscv.best_params_))
	print('\n')
	print('optimize n score:  {}'.format(knn_gscv.best_score_))
	print('----------------------------------------------------------\n')
	print('-----------------------------------------------------------------------\n\n\n')
	
	#knn for selected  feature of dataset
	select_feature = SelectKBest(chi2, 5).fit(x_train, y_train)
	selected_features_data = pd.DataFrame({'Feature':list(x_train.columns),'Scores':select_feature.scores_})
	X_chi = select_feature.transform(X)
	x_train_chi = select_feature.transform(x_train)
	x_test_chi = select_feature.transform(x_test)
	model = KNeighborsClassifier(n_neighbors=7)
	lr_chi_model = model.fit(x_train_chi,y_train.values.ravel())
	#predict & score
	y_predict = lr_chi_model.predict(x_test_chi)
	score=lr_chi_model.score(x_test_chi, y_test)      
    #– Cross validation
	model_cv = KNeighborsClassifier(n_neighbors=7)
	cv_scores = cross_val_score(model_cv, X_chi , Y, cv=5)
        
	"""k_range = range(1, 30)
	k_scores = []
	for k in k_range:
		knn = KNeighborsClassifier(n_neighbors=k)
		scores = cross_val_score(knn, X_chi, Y, cv=5, scoring='accuracy')
		k_scores.append(scores.mean())
	plt.plot(k_range, k_scores)
	plt.xlabel('Value of K for KNN')
	plt.ylabel('Cross-Validated Accuracy')
	plt.show()"""
	#grid search
	model2 = KNeighborsClassifier()
	param_grid = {'n_neighbors': np.arange(1, 30)}
	knn_gscv = GridSearchCV(model2, param_grid, cv=5)
	knn_gscv.fit(X_chi, Y)
	#all values 
	print('\n------------------------KNN for optimal features-----------------------\n')
	print('----------------------score for n=7 ----------------------\n')
	print('accuracy: {}'.format(score))
	print('----------------------------------------------------------\n')
	print('------------cross validationscore for n=7 cv=5------------\n')
	print(cv_scores)
	print('\n')
	print('cv_scores mean: {}'.format(np.mean(cv_scores)))
	print('----------------------------------------------------------\n')
	print('--------------------optimal n for KNN---------------------\n')
	print('optimize n for knn:  {}'.format(knn_gscv.best_params_))
	print('\n')
	print('optimize n score:  {}'.format(knn_gscv.best_score_))
	print('----------------------------------------------------------\n')
	print('-----------------------------------------------------------------------\n\n\n')
	
	
	
	
	#ANN model futures all fudataset
	scaler = StandardScaler()
	scaler.fit(x_train)
	X_train = scaler.transform(x_train)
	X_test = scaler.transform(x_test)
	mlp = MLPClassifier(hidden_layer_sizes=(9, 10, 2), max_iter=1000)
	mlp.fit(X_train, y_train.values.ravel())
	predictions = mlp.predict(X_test)
	score=mlp.score(X_test, y_test)
	print('\n-------------------------ANN for all features--------------------------\n')
	print('accuracy: {}'.format(score))
	print('-----------------------------------------------------------------------\n\n\n')
	
	#ANN model optimal all fudataset
	scaler.fit(x_train_chi)
	X_train = scaler.transform(x_train_chi)
	X_test = scaler.transform(x_test_chi)
	mlp = MLPClassifier(hidden_layer_sizes=(5,10, 2), max_iter=1000)
	mlp.fit(X_train, y_train.values.ravel())
	predictions = mlp.predict(X_test)
	score=mlp.score(X_test, y_test)
	print('\n------------------------ANN for optimal features------------------------\n')
	print('accuracy: {}'.format(score))
	print('-----------------------------------------------------------------------\n\n\n')
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
    




if __name__== "__main__":
	main()

