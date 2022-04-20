# Binary Classification Assignment

## EDA (EDA.ipynb)
The dataset has 57 features and there aren't any Null/NaNs in the features, By looking at the Density plots for the features we can conclude that:  
- All the features are Unimodal
- Right Skewed
- Mean is greater than Median for all features
- Features have high number of zeros
- The Class Distribution is around 60:40 :: Negative:Positive

### Feature Selection
#### Univariate Feature Selection (Statistic based Selection)
1. I conducted Multicollinearity test using Variance inflation factor (VIF) and Pearson Correlation with Heatmaps, I observed that some features are highly correlated for example X34 and X32 also X32 is highly correlated with X36 and X40. So for further analysis I have dropped X34.
2. I Conducted ANOVA f-test to calculate the ratio between the variance values of distribution of feature values coming from positive and negative class. We remove those features that are independent of the target variable. 
3. Using Mutual Information as a measure to guage the importance of a feature with the target variables top features are selected.
#### Recursive Feature Elimination
1. Given an estimator that assigns weights to the features we can eliminate the least important features in a recursive fashion. I have used Logistic Regression and Support Vector Classifier with Linear kernel. 
2. We can also use these estimators with l1 penalty because we know that when we Regularize with l1 penalty the estimator assigns zero weight for the least important feature.
#### Sequential Feature Selection
1. This is a greedy procedure that iteratively finds the best new feature to adds it to the set of selected features, I have used K-Nearest Neighbor Classifier and 2 tree based estimators.
2. In the forward direction it starts with zero selected features and performing (Number of features to be selected * K cross validation fits) to select the required Number of features. I have used Decision Tree Classifier and Extra Tree Classifier in the forward direction.
#### Dimensionality Reduction (PCA)
1. Principal component analysis (PCA) has been used to reduce the dimension using the Singular Value Decomposition method. The returned Principal components explain close to 100 percent of the total variance in the dataset.

## Modelling
An ensembe Voting Classifier has been created in the model.py. This can be summarized in the image below:  

![Model Image](https://github.com/rietesh/arya_assignment/blob/master/Model.png?raw=true)  

Model can be created by passing Number of features to be selected and Classifiers either a list of classifiers or a single classifier

### Usage Example
``` 
train = pd.read_csv('./training_set.csv')
train = train.drop(columns=['Unnamed: 0','X34'])
X = train.drop(columns=['Y'])
y = train['Y']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=4535, stratify=y, shuffle=True)
clf = Model(30, LogisticRegression())
clf.fit(X_train, y_train)

test_pred = clf.predict(X_test) 
```
Or we can pass in a list of estimators

```
classifiers = [SVC(kernel='linear',probability=True), SVC(kernel='rbf',probability=True), SVC(kernel='sigmoid',probability=True),
               LogisticRegression(),GaussianNB(), ExtraTreeClassifier(), DecisionTreeClassifier(), LogisticRegression()]

clf2 = Model(30, classifiers)
clf2.fit(X_train, y_train)

test_pred = clf2.predict(X_test)
```

## Note
- Submission.csv has a column of predictions obtanied for the test_set.csv
- A list of dependencies/libraries can be found in the Requirements.txt
- Model performance and analysis can be run from modelling.ipynb file, The plot to compare Logistic Regression and Linear SVC might take longer time(~ 10 min)
