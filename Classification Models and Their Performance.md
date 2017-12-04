---
title: Classification Models and Their Performance
notebook: Classification Models and Their Performance.ipynb
nav_include: 3
---

## Contents
{:.no_toc}
*  
{: toc}



```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')
```


## Load data



```python
df_train = pd.read_csv("data/ADNIMERGE_train.csv")
df_test = pd.read_csv("data/ADNIMERGE_test.csv")
```




```python
X_train = df_train.drop(['RID', 'DX_bl'], axis=1).copy()
y_train = df_train['DX_bl'].copy()
X_test = df_test.drop(['RID', 'DX_bl'], axis=1).copy()
y_test = df_test['DX_bl'].copy()
```




```python
def score(model, X_train, y_train, X_test, y_test):
    train_acc = model.score(X_train,y_train)
    test_acc = model.score(X_test,y_test)
    test_class0 = model.score(X_test[y_test==0], y_test[y_test==0])
    test_class1 = model.score(X_test[y_test==1], y_test[y_test==1])
    test_class2 = model.score(X_test[y_test==2], y_test[y_test==2])
    return pd.Series([train_acc, test_acc, test_class0, test_class1, test_class2],
                    index = ['Train accuracy', 'Test accuracy', 
                             "Test accuracy CN", "Test accuracy CI", "Test accuracy AD"])
```


## Logistic Regression

We tested 6 kinds of logistic regression, logistic regression with l1 penalty, logistic regression with l2 penalty, unweighted logistic regression, weighted logistic regression, one-vs-rest logistic regression and multinomial logistic regression. We chose the best parameters with cross validation. We found that unless we used weighted logistic regression, we need a large regularization term. However, the accuracy of weighted logistic regression is very low compared to the others. That indicates that we have too many variables.



```python
#l1
log_l1 = LogisticRegressionCV(penalty = 'l1', solver = 'liblinear')
log_l1.fit(X_train,y_train)

#l2
log_l2 = LogisticRegressionCV(penalty = 'l2')
log_l2.fit(X_train,y_train)

#Unweighted logistic regression
unweighted_logistic = LogisticRegressionCV()
unweighted_logistic.fit(X_train,y_train)

#Weighted logistic regression
weighted_logistic = LogisticRegressionCV(class_weight='balanced')
weighted_logistic.fit(X_train,y_train)

#ovr
log_ovr = LogisticRegressionCV(multi_class = 'ovr')
log_ovr.fit(X_train,y_train)

#multinomial
log_multinomial = LogisticRegressionCV(multi_class = 'multinomial', solver = 'newton-cg')
log_multinomial.fit(X_train,y_train)

print("Regularization strength: ")
print("-------------------------")
print("Logistic regression with l1 penalty:", log_l1.C_[0])
print("Logistic regression with l2 penalty:", log_l2.C_[0])
print("Unweighted logistic regression: ", unweighted_logistic.C_[0])
print("Weighted logistic regression: ", weighted_logistic.C_[0])
print("OVR logistic regression: ", log_ovr.C_[0])
print("Multinomial logistic regression: ", log_multinomial.C_[0])
```


    Regularization strength: 
    -------------------------
    Logistic regression with l1 penalty: 2.78255940221
    Logistic regression with l2 penalty: 0.35938136638
    Unweighted logistic regression:  0.35938136638
    Weighted logistic regression:  1291.54966501
    OVR logistic regression:  0.35938136638
    Multinomial logistic regression:  21.5443469003




```python
#Computing the score on the train set - 
print("Training accuracy")
print("-------------------------------------------------")
print('Logistic Regression with l1 penalty train Score: ',log_l1.score(X_train, y_train))
print('Logistic Regression with l2 penalty train Score: ',log_l2.score(X_train, y_train))
print('Unweighted Logistic Regression with train Score: ',unweighted_logistic.score(X_train, y_train))
print('Weighted Logistic Regression train Score: ',weighted_logistic.score(X_train, y_train))
print('OVR Logistic Regression train Score: ',log_ovr.score(X_train, y_train))
print('Multinomial Logistic Regression train Score: ',log_multinomial.score(X_train, y_train))

print('\n')

#Computing the score on the test set - 
print("Test accuracy")
print("-------------------------------------------------")
print('Logistic Regression with l1 penalty test Score: ',log_l1.score(X_test, y_test))
print('Logistic Regression with l2 penalty test Score: ',log_l2.score(X_test, y_test))
print('Unweighted Logistic Regression with test Score: ',unweighted_logistic.score(X_test, y_test))
print('Weighted Logistic Regression test Score: ',weighted_logistic.score(X_test, y_test))
print('OVR Logistic Regression test Score: ',log_ovr.score(X_test, y_test))
print('Multinomial Logistic Regression test Score: ',log_multinomial.score(X_test, y_test))
```


    Training accuracy
    -------------------------------------------------
    Logistic Regression with l1 penalty train Score:  0.829307568438
    Logistic Regression with l2 penalty train Score:  0.618357487923
    Unweighted Logistic Regression with train Score:  0.618357487923
    Weighted Logistic Regression train Score:  0.484702093398
    OVR Logistic Regression train Score:  0.618357487923
    Multinomial Logistic Regression train Score:  0.840579710145
    
    
    Test accuracy
    -------------------------------------------------
    Logistic Regression with l1 penalty test Score:  0.783950617284
    Logistic Regression with l2 penalty test Score:  0.592592592593
    Unweighted Logistic Regression with test Score:  0.592592592593
    Weighted Logistic Regression test Score:  0.425925925926
    OVR Logistic Regression test Score:  0.592592592593
    Multinomial Logistic Regression test Score:  0.746913580247




```python
l1_score = score(log_l1, X_train, y_train, X_test, y_test)
l2_score = score(log_l2, X_train, y_train, X_test, y_test)
weighted_score = score(weighted_logistic, X_train, y_train, X_test, y_test)
unweighted_score = score(unweighted_logistic, X_train, y_train, X_test, y_test)
ovr_score = score(log_ovr, X_train, y_train, X_test, y_test)
multi_score = score(log_multinomial, X_train, y_train, X_test, y_test)
```


## Discriminant Analysis

We performed normalization on continuous predictors and used Linear Discriminant Analysis (LDA) and Quadratic Discriminant Analysis (QDA) as our models. 



```python
cols_continuous = ['APOE4', 'CSF_ABETA', 'CSF_TAU',  'CSF_PTAU', 
                   'FDG', 'FDG_slope', 'AV45', 'AV45_slope',
                   'ADAS13', 'ADAS13_slope', 'MMSE', 'MMSE_slope',
                   'RAVLT_immediate', 'RAVLT_immediate_slope', 'RAVLT_learning',
                   'RAVLT_learning_slope', 'RAVLT_forgetting', 'RAVLT_forgetting_slope',
                   'RAVLT_perc_forgetting', 'RAVLT_perc_forgetting_slope', 'MOCA',
                   'MOCA_slope', 'EcogPtMem', 'EcogPtMem_slope', 'EcogPtLang',
                   'EcogPtLang_slope', 'EcogPtVisspat', 'EcogPtVisspat_slope',
                   'EcogPtPlan', 'EcogPtPlan_slope', 'EcogPtOrgan', 'EcogPtOrgan_slope',
                   'EcogPtDivatt', 'EcogPtDivatt_slope', 'EcogSPMem', 'EcogSPMem_slope',
                   'EcogSPLang', 'EcogSPLang_slope', 'EcogSPVisspat',
                   'EcogSPVisspat_slope', 'EcogSPPlan', 'EcogSPPlan_slope', 'EcogSPOrgan',
                   'EcogSPOrgan_slope', 'EcogSPDivatt', 'EcogSPDivatt_slope', 'FAQ',
                   'FAQ_slope', 'Ventricles', 'Ventricles_slope', 'Hippocampus',
                   'Hippocampus_slope', 'WholeBrain', 'WholeBrain_slope', 'Entorhinal',
                   'Entorhinal_slope', 'Fusiform', 'Fusiform_slope', 'MidTemp',
                   'MidTemp_slope', 'ICV', 'ICV_slope']

X_train_std = X_train.copy()
X_test_std = X_test.copy()
for i in cols_continuous:
    col_mean = np.mean(X_train_std[i])
    col_sd = np.std(X_train_std[i])
    if col_sd < 1e-10*col_mean:
        X_train_std.loc[i] = (X_train_std[i]-col_mean)/col_sd
        X_test_std.loc[i] = (X_test_std[i]-col_mean)/col_sd
```




```python
lda = LinearDiscriminantAnalysis()
qda = QuadraticDiscriminantAnalysis()

lda.fit(X_train_std,y_train)
qda.fit(X_train_std,y_train)

print("Training accuracy")
print("------------------")
print('LDA Train Score: ',lda.score(X_train_std,y_train))
print('QDA Train Score: ',qda.score(X_train_std,y_train))

print('\n')

print("Test accuracy")
print("------------------")
print('LDA Test Score: ',lda.score(X_test_std,y_test))
print('QDA Test Score: ',qda.score(X_test_std,y_test))
```


    Training accuracy
    ------------------
    LDA Train Score:  0.85346215781
    QDA Train Score:  0.837359098229
    
    
    Test accuracy
    ------------------
    LDA Test Score:  0.796296296296
    QDA Test Score:  0.70987654321




```python
lda_score = score(lda, X_train_std, y_train, X_test_std, y_test)
qda_score = score(qda, X_train_std, y_train, X_test_std, y_test)
```


## K-Nearest Neighbours

The optimal number of neighbours is 37, which is a relatively large number considering that we only have 783 observations. The accuracy is not satisfactory as well.



```python
max_score = 0
max_k = 0 

for k in range(1,60):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn_val_score = cross_val_score(knn, X_train, y_train).mean()
    if knn_val_score > max_score:
        max_k = k
        max_score = knn_val_score
        
knn = KNeighborsClassifier(n_neighbors = max_k)
knn.fit(X_train,y_train)

print("Optimal number of neighbours: ", max_k)
print('KNN Train Score: ', knn.score(X_train,y_train))
print('KNN Test Score: ', knn.score(X_test,y_test))

knn_score = score(knn, X_train, y_train, X_test, y_test)
```


    Optimal number of neighbours:  37
    KNN Train Score:  0.566827697262
    KNN Test Score:  0.592592592593


## Decision Tree

We used 5-fold cross validation to find the optimal depth for the decision tree. The optimal depth is 4.



```python
depth = []
for i in range(3,20):
    dt = DecisionTreeClassifier(max_depth=i)
    # Perform 5-fold cross validation 
    scores = cross_val_score(estimator=dt, X=X_train, y=y_train, cv=5, n_jobs=-1)
    depth.append((i, scores.mean(), scores.std())) 
depthvals = [t[0] for t in depth]
cvmeans = np.array([t[1] for t in depth])
cvstds = np.array([t[2] for t in depth])
max_indx = np.argmax(cvmeans)
md_best = depthvals[max_indx]
print('Optimal depth:',md_best)
dt_best = DecisionTreeClassifier(max_depth=md_best)
dt_best.fit(X_train, y_train).score(X_test, y_test)
dt_score = score(dt_best, X_train, y_train, X_test, y_test)
```


    Optimal depth: 4




```python
print('Decision Tree Train Score: ', dt_best.score(X_train,y_train))
print('Decision Tree Test Score: ', dt_best.score(X_test,y_test))
```


    Decision Tree Train Score:  0.818035426731
    Decision Tree Test Score:  0.753086419753


## Random Forest

We used `GridSearchCV` to find the optimal number of trees and tree depth. We then used the optimal value to perform random forest classification.



```python
trees = [2**x for x in range(8)]  # 1, 2, 4, 8, 16, 32, ...
depth = [2, 4, 6, 8, 10]
parameters = {'n_estimators': trees,
              'max_depth': depth}
rf = RandomForestClassifier()
rf_cv = GridSearchCV(rf, parameters)
rf_cv.fit(X_train, y_train)
best_score = np.argmax(rf_cv.cv_results_['mean_test_score'])
result = rf_cv.cv_results_['params'][best_score]
opt_depth = result['max_depth']
opt_tree = result['n_estimators']
print("Optimal number of trees {}, tree depth: {}".format(opt_tree, opt_depth))
rf = RandomForestClassifier(n_estimators=opt_tree, max_depth=opt_depth)
rf.fit(X_train, y_train)
print('\n')
print('Random Forest Train Score: ', rf.score(X_train,y_train))
print('Random Forest Test Score: ', rf.score(X_test,y_test))
rf_score = score(rf, X_train, y_train, X_test, y_test)
```


    Optimal number of trees 32, tree depth: 6
    
    
    Random Forest Train Score:  0.9307568438
    Random Forest Test Score:  0.820987654321


## AdaBoost

We used the optimal tree depth found by cross validation in the decision tree classifier, and performed `GridSearchCV` to find the optimal number of trees and learning rate.



```python
trees = [2**x for x in range(6)]  # 1, 2, 4, 8, 16, 32, ...
learning_rate = [0.1, 0.5, 1, 5]
parameters = {'n_estimators': trees,
              'learning_rate': learning_rate}
ab = AdaBoostClassifier(DecisionTreeClassifier(max_depth=md_best))
ab_cv = GridSearchCV(ab, parameters)
ab_cv.fit(X_train, y_train)
best_score = np.argmax(ab_cv.cv_results_['mean_test_score'])
result = ab_cv.cv_results_['params'][best_score]
opt_learning_rate = result['learning_rate']
opt_tree = result['n_estimators']
print("Optimal number of trees {}, learning rate: {}".format(opt_tree, opt_learning_rate))
ab = AdaBoostClassifier(DecisionTreeClassifier(max_depth=md_best), n_estimators=opt_tree,
                       learning_rate=opt_learning_rate)
ab.fit(X_train, y_train)
print('\n')
print('AdaBoost Train Score: ', ab.score(X_train,y_train))
print('AdaBoost Test Score: ', ab.score(X_test,y_test))
ab_score = score(ab, X_train, y_train, X_test, y_test)
```


    Optimal number of trees 32, learning rate: 1
    
    
    AdaBoost Train Score:  0.971014492754
    AdaBoost Test Score:  0.722222222222


## Performance Summary



```python
score_df = pd.DataFrame({'Logistic Regression with l1': l1_score, 
                         'Logistic Regression with l2': l2_score,
                         'Weighted logistic': weighted_score,
                         'Unweighted logistic': unweighted_score,
                         'OVR': ovr_score,
                         'Multinomial': multi_score,
                         'KNN': knn_score,
                         'LDA': lda_score,
                         'QDA': qda_score,
                         'Decision Tree': dt_score,
                         'Random Forest': rf_score,
                         'AdaBoost': ab_score})
score_df
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AdaBoost</th>
      <th>Decision Tree</th>
      <th>KNN</th>
      <th>LDA</th>
      <th>Logistic Regression with l1</th>
      <th>Logistic Regression with l2</th>
      <th>Multinomial</th>
      <th>OVR</th>
      <th>QDA</th>
      <th>Random Forest</th>
      <th>Unweighted logistic</th>
      <th>Weighted logistic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Train accuracy</th>
      <td>0.971014</td>
      <td>0.818035</td>
      <td>0.566828</td>
      <td>0.853462</td>
      <td>0.829308</td>
      <td>0.618357</td>
      <td>0.840580</td>
      <td>0.618357</td>
      <td>0.837359</td>
      <td>0.930757</td>
      <td>0.618357</td>
      <td>0.484702</td>
    </tr>
    <tr>
      <th>Test accuracy</th>
      <td>0.722222</td>
      <td>0.753086</td>
      <td>0.592593</td>
      <td>0.796296</td>
      <td>0.783951</td>
      <td>0.592593</td>
      <td>0.746914</td>
      <td>0.592593</td>
      <td>0.709877</td>
      <td>0.820988</td>
      <td>0.592593</td>
      <td>0.425926</td>
    </tr>
    <tr>
      <th>Test accuracy CN</th>
      <td>0.380952</td>
      <td>0.476190</td>
      <td>0.023810</td>
      <td>0.619048</td>
      <td>0.571429</td>
      <td>0.000000</td>
      <td>0.642857</td>
      <td>0.000000</td>
      <td>0.833333</td>
      <td>0.619048</td>
      <td>0.000000</td>
      <td>0.833333</td>
    </tr>
    <tr>
      <th>Test accuracy CI</th>
      <td>0.870968</td>
      <td>0.838710</td>
      <td>0.989247</td>
      <td>0.849462</td>
      <td>0.860215</td>
      <td>0.924731</td>
      <td>0.763441</td>
      <td>0.924731</td>
      <td>0.623656</td>
      <td>0.924731</td>
      <td>0.924731</td>
      <td>0.172043</td>
    </tr>
    <tr>
      <th>Test accuracy AD</th>
      <td>0.740741</td>
      <td>0.888889</td>
      <td>0.111111</td>
      <td>0.888889</td>
      <td>0.851852</td>
      <td>0.370370</td>
      <td>0.851852</td>
      <td>0.370370</td>
      <td>0.814815</td>
      <td>0.777778</td>
      <td>0.370370</td>
      <td>0.666667</td>
    </tr>
  </tbody>
</table>
</div>



Based on the above summary, AdaBoost has a very high train accuracy which is close to 1(0.971014). Random Forest Classifier ranks second on the train accuracy(0.930757) and it also has the highest accuracy(0.820988) on the test set. LDA has the second highest accuracy(0.796296) and logistic regression with l1 regularization is the third highest(0.783951).

For classifying `CN` patients, both QDA and weighted logistic regression have the highest test accuracy(0.833333), so they performed the best for determining Cognitive Normal patients. However, logistic regression with l2 regularization and unweighted logistic regression have zero accuracy on classifying `CN` patients. Since both of them have very high accuracy on `CI` but low accuracy on `AD`, we think these two models probably classified all the `CN` patients into `AD` which leads to zero accuracy on `CN` and low accuracy on `AD`.

KNN has the highest test accuracy(0.989247) on diagnosing `CI` patients. Logistic regression with l2 regularization, OvR logistic regression and random forest all reached 0.9 accuracy on diagnosing `CI` patients.

Since we focus on the diagnosis of Alzheimer's disease, we are more concerned about the test accuracy on `AD` patients. Both Decision Tree and LDA have the highest test accuracy(0.888889) on `AD` patients. Logistic regression with l1 regularization and multinomial logistic regression both reached test accuracy of over 0.85 on the classification of `AD`.

To conclude, Decision Tree and LDA performed the best if we only concern about diagnosing `AD` patients. However, Random Forest has the best performance overall. 
