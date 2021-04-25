#Model 1.2
#About this model : Approximate coefficients in wavelet domain (dks) -> SWDA for feature selection -> QDA classifier
#Training Set(unbalanced): 60 target, 180 non target, 25:75 ratio. Testing set : 26 target, 26 non-target
import numpy as np
import matplotlib.pyplot as plt 
from pandas import read_csv
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.utils import class_weight
import time



def visualize(X_train_tar,X_train_nontar,X,y,dim):   #method for visualization post dimensionality reduction. features are projected onto 2 most discriminative directions (eigen vectors)

	def S_withinclass(X_train,class_mean):  #calculate within class distances
		length=np.shape(X_train)[0]
		S=np.zeros((dim,dim))
		for i in range(length):
			X_features=X_train[i,0:dim]
			S +=np.dot(np.transpose(X_features-class_mean), (X_features-class_mean))
		return(S)

	def S_betweenclass(class_mean,mean_data,N_class): #calculate between class distances
		class_mean=np.array(class_mean)
		mean_data=np.array(mean_data)
		S_B=np.zeros((dim,dim))
		#print(class_mean-mean_data)
		S_B=np.dot(np.transpose(class_mean-mean_data), (class_mean-mean_data))
		S_B=N_class*np.array(S_B)
		return(S_B)

	mean_vectors_tar = []
	mean_vectors_nontar=[]
	mean_vectors_tar.append(np.mean(X_train_tar[:,:], axis=0)) #mean of target and non-target classes
	mean_vectors_nontar.append(np.mean(X_train_nontar[:,:],axis=0))
	print(np.shape(mean_vectors_tar))
	mean_data=[]
	mean_data.append(np.mean((X),axis=0))

	S_tar=S_withinclass(X_train_tar,mean_vectors_tar) 
	S_nontar=S_withinclass(X_train_nontar,mean_vectors_nontar)

	S_W=S_tar+S_nontar

	N_tar=np.shape(X_train_tar)[0]
	N_nontar=np.shape(X_train_nontar)[0]
	S_Btar=S_betweenclass(mean_vectors_tar,mean_data,N_tar)
	S_Bnontar=S_betweenclass(mean_vectors_nontar,mean_data,N_nontar)
	S_B=S_Btar+S_Bnontar

	tar_cov=S_Btar/(N_tar-1)  #displaying covariance matrices of both classes as a justification for using QDA 
	nontar_cov=(S_Bnontar/(N_nontar-1))
	print((S_Btar/(N_tar-1)),'tar covariance')
	print((S_Bnontar/(N_nontar-1)),'non tar covariance') 
	print("error in covariance matrices of both classes!")
	print(tar_cov-nontar_cov)

	eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))  # solving for Fisher's Criteria 

	for i in range(len(eig_vals)):
    		eigvec_sc = eig_vecs[:,i].reshape(dim,1)   
    	
	eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
	eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)  # sorting eigen pairs in descending order to obtain the axes for dimensionality reduction
	#for i in eig_pairs:
    		#print(i[0])

	W1=eig_pairs[0][1].reshape(dim,1)  #the eigen vector corresponding to the highest eigen value  
	W2=eig_pairs[1][1].reshape(dim,1)  
	W=np.concatenate((W1,W2),axis=1)

	X_lda=np.matmul(X,W.real) #projecting data points onto the eigen vectors. X_lda is the reduced feature space after transformation
	
	colors=("red","blue")
	targets=plt.scatter(X_lda[0:N_tar,0],X_lda[0:N_tar,1],c=colors[0]) #creating scatter plot of target and non-target
	nontargets=plt.scatter(X_lda[N_tar:(np.shape(X)[0]),0],X_lda[N_tar:np.shape(X)[0],1],c=colors[1])
	plt.legend((targets,nontargets),('targets','nontargets'))
	plt.title('Projecting Training Dataset after applying FLD')
	plt.xlabel('LD1')
	plt.ylabel('LD2')
	plt.show()


def LDA(X,y,test_samples,y_true):
	
	clf=LinearDiscriminantAnalysis(solver='eigen',store_covariance=True) #LDA method 
	clf.fit(X,y) #training classifier
	cov=clf.covariance_
	prior=clf.priors_ 
	y_pred=clf.predict(test_samples)  #testing classifier accuracy
	scores=clf.score(test_samples,y_true) #classifier accuracy score
	trans=clf.transform(X) #for dimensionality reduction only
	return y_pred,scores,trans


def QDA(X,y,test_samples,y_true):

	clf=QuadraticDiscriminantAnalysis(store_covariance=True) #QDA method
	clf.fit(X,y)	#train QDA classifier
	y_pred=clf.predict(test_samples) 
	scores=clf.score(test_samples,y_true)
	prob=clf.predict_proba(test_samples)
	cov=clf.covariance_
	return y_pred,scores,cov
	

def validate(y_true,y_pred):

	C=sklearn.metrics.confusion_matrix(y_true, y_pred, labels=None, sample_weight=None)  #evaluating confusion matrices for validation 
	TN=C[0][0] #pred non targets when actually non targets (True Negatives)
	TP=C[1][1]  #True Positives 
	FN=C[1][0]  #False Negatives
	FP=C[0][1]  #False Positives
	tot=(np.shape(y_pred)[0]*1.0)
	overall_accuracy=(TP+TN)/tot
	misclassification_rate=(FP+FN)/tot

	unique_true, counts_true = np.unique(y_true, return_counts=True)
	print(np.asarray((unique_true,counts_true)).T)
	true_tar_count=(np.asarray((unique_true, counts_true)).T)[1][1]   #obtaining true target and non-target counts
	true_nontar_count=tot-true_tar_count
	#print(true_tar_count)

	unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
	print(np.asarray((unique_pred,counts_pred)).T)
	pred_tar_count=(np.asarray((unique_pred, counts_pred)).T)[1][1]
	pred_nontar_count=tot-pred_tar_count 
	
	Sensitivity=TP/(true_tar_count*1.0)    #evaluating classification parametrs - sensitivity,specificity, precision
	#Sensitivity=TP/((TP+FN)*1.0)
	False_positive_rate=FP/(true_nontar_count*1.0)
	Specificity=TN/(true_nontar_count*1.0)
	Precision=TP/(pred_tar_count*1.0)
	Prevalence=true_tar_count/(tot*1.0)

	print(C,'confusion matrix')
	print(overall_accuracy,'overall accuracy')
	print(misclassification_rate,'misclassification rate')
	print(Sensitivity,'sensitivity-true positive rate- TP/actual targets')
	print(Specificity,'Specificity-true negatives rate- TN/actual nontargets')
	print(Precision,'Precision-TP/pred target num')

	print(TN,'True Negatives - predicted non targets when actually non targets')
	print(FP,'False positives - predicted targets but actually non targets')
	print(FN,'False negatives - predicted non targets but actually targets')
	print(TP,'True positives - predicted targets when actually targets')

train_set=np.array(read_csv('/Users/raksharamesh/Documents/Rak_Classifiermodels/SWDA_unbal/swdafeat_unbal_festures.csv',header=None)) # path to training set post SWDA. 
#SWDA is performed in MATLAB and only significant features are stored in training set out of the entire feature set

dkstar_train=train_set[0:60,:]
dksnontar_train=train_set[60:240,:]
X=train_set[:,0:(np.shape(train_set)[1]-1)]
#X=X[0:120,:]  # if balanced training set is required 
#X=train_set[:,0:(np.shape(train_set)[1]-1)]
y=train_set[:,((np.shape(train_set)[1])-1)] # y contains labels of training dataset 
#y=y[0:120]

test_set= np.array(read_csv('/Users/raksharamesh/Documents/Rak_Classifiermodels/SWDA_unbal/test_set.csv',header=None))

#test_set=np.array(read_csv('/Users/raksharamesh/Downloads/test_samples_openvibe.csv',header=None))
col_name=(8, 28, 29, 31, 32, 33, 34, 43, 45, 46, 48, 50, 55, 56, 57, 59, 60, 61, 62, 63, 68, 81, 83, 84) #picking SWDA selected column numbers from test set.  column 84 contains true labels
print(test_set,'test set')
test_set=test_set[:,col_name]
test_samples=test_set[:,0:np.shape(test_set)[1]-1]
print(test_samples,"TEST SAMPLES")
print(np.shape(test_samples),"test dim")
y_true=test_set[:,((np.shape(test_set)[1]-1))]  #true labels of test set

visualize(dkstar_train[:,0:(np.shape(dkstar_train)[1]-1)],dksnontar_train[:,0:(np.shape(dksnontar_train)[1]-1)],X,y,23) #Pass No of features 

#y_predLDA,scoresLDA,transLDA=LDA(X,y,test_samples,y_true)
y_predQDA,scoresQDA,cov=QDA(X,y,test_samples,y_true)  #using QDA for unbalanced training set due to dissimilar covariance matrices for both classes 
#print(y_predLDA,'predicted labels')
#validate(y_true,y_predLDA)
validate(y_true,y_predQDA)
#print(scoresLDA,'scores from LDA') 
print(scoresQDA,'scores from QDA')
#print(y_predLDA,'predicted labels from LDA') 
print(y_predQDA,'predicted labels from QDA') 
print(y_true,'actual labels of test data')




