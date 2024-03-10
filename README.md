# Heart-Disease-Prediction

## Introduction

When the heart cannot pump enough blood to meet the body's needs, it develops heart failure (HF), a medical ailment. Worldwide incidences of heart disease are rising daily at an unprecedented and exponential rate. Careful diagnosis and implementation of a healthy lifestyle can eliminate many cardiovascular diseases. Machine learning models can improve the detection and manage high-risk cardiovascular patients. The healthcare industry relies on machine learning models to predict heart failure cases. Medical examiners and cardiac practitioners can expect heart disease using a statistical model based on patient health history.

The Sheffield Hallam University clinic team has asked to explore different programming and analytics techniques to analyse and evaluate heart failure model performance to make informed decisions on patient survival. For analysis, the UCI Dataset includes vital clinical characteristics of a patient that are crucial for determining the presence of cardiac disease. The objective is to categorise or forecast whether a patient will experience a circumstance where heart failure may cause death. With numerous numerical and category features, this is a binary classification issue.

Six distinct Machine Learning Classifier Models, including Naive Bayes, Logistic Regression, Support Vector Machine, Random Forest Classifier, K-Nearest Neighbour, and Multi-Layer Perceptron Neural Network, are utilised to construct the prediction model.

In this project, the following steps are being performed for classification and prediction:

<ul>
  <li>Exploratory Data Analysis</li>
  <li>Classification I</li>
  <li>Classification II</li>
  <li>Feature Selection</li>
  <li>Classification III</li>
</ul>

### Exploratory Data Analysis 

For exploratory data analysis, to determine a patient's heart disease, the heart failure clinical records dataset from the UCI Machine Learning Repository is employed. It consists of observations of 299 patients for 13 different attributes. The various attributes include:

a) **Age:** The patient's age in years. \
b) **Anaemia:** Reduced levels of haemoglobin or red blood cells (0:Reduced or 1:Normal) \
c) **creatinine_phosphokinase:** Blood CPK enzyme concentration (mcg/L) \
d) **Diabetes:** If the patient suffers from diabetes (0:No or 1:Yes) \
e) **ejection_fraction:** The proportion of blood that leaves the heart with each contraction (percentage) \
f) **high_blood_pressure:** If the patient has high blood pressure(0:No or 1:Yes) \
g) **platelets:** The number of platelets per millilitre of blood. \
h) **serum_creatinine:** Blood serum creatinine level in milligrammes per deciliter \
i) **serum_sodium:** Blood sodium concentration (mEq/L) in the serum \
j) **sex:** The patient's biological sex (0:Female or 1:Male) \
k) **smoking:** In the event that the patient smokes (0:No or 1: Yes) \
l) **time:** The amount of time to follow up in days \
m) **death_event:** If the patient lived until the conclusion of the follow-up period, this is a death event (0: No or 1:Yes )

Missing/Null values can bias the machine learning models' results, thus reducing the accuracy.

Therefore, it is necessary to apply strategies to deal with the missing values/null values before the dataset is passed to the machine learning or deep learning framework. There are no null values present in the dataset. It is evident as there are no data points in the heat map.

**IMAGE 1**
                      
Categorical and numerical features make up the attributes of the dataset. Categorical data are data types that can be stored and recognised depending on the names or labels given to them. Data expressed as numbers instead of in any linguistic or descriptive form is compared to numerical data.

<ul>
  <li>Anaemia, diabetes, high blood pressure, sex, smoking, and DEATH EVENT are <b>categorical</b> features.</li>
  <li>Age, creatinine phosphokinase, ejection fraction, platelets, serum creatinine, serum sodium and time are <b>numerical</b> features.</li>
</ul>
 
Multiple python functions are created to carry out the descriptive statistical analysis of the numerical features. The following statistical information has been found:

**IMAGE 2**

The distribution of categorical and numerical features are studied by plotting the charts. It is found that:

**IMAGE 3**

<ul>
  <li>All the categorical features are <b>Normally Distributed.</b></li>
</ul>
 
**IMAGE 4**

<ul>
  <li>The data distribution for <b>age</b>, <b>ejection fraction</b>, <b>creatinine phosphokinase</b> and <b>serum creatinine</b> is rightly or positively skewed.</li>
  <li><b>Platelets</b> and <b>Serum_Sodium</b> are almost normally distributed.</b></li>
</ul>

<b>Time's</b> data distribution is similar to a typical <b>Time Series Analysis</b> graph with irregularities.

Different visualisations are done, including Categorical Features vs Target Variable (DEATH_EVENT) and Numerical Features vs Target Variable (DEATH_EVENT), to understand the dataset better. Moreover, the mean values of all the  features are found for cases of DEATH_EVENT and No DEATH_EVENT.

**IMAGE 5**

To address the overfitting and underfitting of the training model, the initial modelling dataset is divided into training and testing samples/sets. The testing set is a collection of data points that helps determine whether the model can generalise effectively to new or unseen data. The training set is used to train the model.

To make training and testing sets, use the <b>sklearn</b> library's <b>train_test_split()</b> method. The <b>test size</b> determines how much of the dataset is included in the test split, whereas the <b>random_state</b> determines how the data are shuffled before the test split is applied.

After splitting it into training and testing samples, the training dataset is normalised. It is done because attributes or features that are measured at various scales do not contribute evenly to the model fitting and model learnt function and may ultimately result in <b>bias</b>. Therefore, <b>MinMax Scaling</b> is typically applied before model fitting.

### Classification I

The new normalised test sets are used for classification using Machine Learning Algorithms such as <b>Naive Bayes</b>, <b>Logistic Regression</b>, <b>Support Vector Machine</b>, <b>Random Forest Classifier</b>, <b>K-Nearest Neighbour</b> and <b>Multi-Layer Perceptron Neural Network</b>. The models are evaluated using the <b>test dataset</b>, and <b>confusion matrices</b> are produced for all the models. 

A confusion matrix is used to visualise and summarise the performance of a classification algorithm. There are different evaluation metrics to understand the model performance outside of the accuracy metric when there is a strong imbalance in test data. 

**Accuracy:** indicates the proportion of accurate predictions among all predictions. A data set that is not balanced is not a valid metric. /
**Precision:** It should ideally be 1 (high) for a good classifier. /
**Recall:**  It should ideally be 1 (high) for a good classifier.

**IMAGE-6**
                                              
**F1 Score:** Metric which considers recall and precision. It is generally described as the harmonic mean of the two.

**IMAGE-7**

The performances of all these models are compared in terms of above mentioned evaluation metrics.

**IMAGE-8**

In **Classification I**, the **Random Forest Classifier** performed the best, whereas the **Support Vector Machine** performed the worst. 

Accuracy of Random Forest Class: 75.00% /
Accuracy of Support Vector Machine: 58.33%

### Classification II 

The target variable “**DEATH_EVENT**” is visualised to investigate the class imbalance problem. It is found that the dataset is unbalanced with low data points(299). The ratio of Death Event Cases to No Death Event Cases is 2:1, and thus the predictions and visualisations will be biased towards No Death Event cases. 

**IMAGE-9**
   
No Death Events/Survived: 203 /
Death Events/Dead: 96

In machine learning classification problems, models will not work well when the training data is unbalanced. Due to the class imbalance, there is a strong bias in favour of the dominant class, which lowers classification accuracy and raises the rate of false negatives.

It is necessary to have a balanced dataset as it considers the same amount of information for predicting each class and gives a better idea of how to respond to test data, resulting in improved classification performance.

#### Method 1: Under-sampling  

In this method, some data is deleted from rows of data of the majority classes. 

**IMAGE-10**

No Death Events/Survived: 203 /
Death Events/Dead: 96

**IMAGE-11**
          
**Limitation:** It is difficult to use when there is no substantial (and relatively equal) data from each target class.

After applying the **Under-sampling** technique on the imbalanced dataset, the new dataset is fed into different models. In this case, the **Random Forest Classifier** performed the best, whereas the **Support Vector Machine** performed the worst.

Accuracy of Random Forest Class: 89.74% /
Accuracy of Support Vector Machine: 46.15%

#### Method 2: SMOTE - Synthetic Minority Oversampling Technique

In this method, new data is generated based on the implications of old data. Current inputs provide distinct input rows with a label depending on what the original data implies rather than deleting or copying the data.

**IMAGE-12**

No Death Events/Survived: 203 /
Death Events/Dead: 203

**IMAGE-13**

After applying the **Under-sampling** technique on the imbalanced dataset, the new dataset is fed into different models. In comparison, the Random Forest Classifier performed better than the Neural Network Model.

Accuracy of Random Forest Class: 84.14% /
Accuracy of Neural Network Model: 50.00%

### Feature Selection

Feature Selection is done to choose a limited set of features dependent on the target variable for use in the model resulting in good performance results. It is achieved by investigating the significance of the features for selection purposes. The distribution of each feature between the target class's two groups (Survived vs. Dead) is first compared using the Mann-Whitney test and Chi-Square test.

**IMAGE-14**

The features are ranked in the most significant order (using P = 0.05).  The feature importance plot is produced from the Random Forest Classifier in III.

**IMAGE-15**                    

The features to be selected are determined using the plot and the results of the statistical tests. **The dependent variable is essentially equally affected by high correlation characteristics, which are more linearly dependent.** Therefore, if there is a strong correlation between two features, one of them may be eliminated. **Time, serum creatinine** and **ejection_fraction** are the most crucial variables.

### Classification III  

Using the features selected in the Feature Selection step, the latest balanced dataset is used for classification models.

**IMAGE-16**  
              
After applying the **Under-sampling** technique on the imbalanced dataset and feature selection, the new dataset is fed into different models. The Support Vector Machine fared poorly, while the **Random Forest Classifier**, **Logistic Regression** and **K Nearest Neighbours** performed well.

Accuracy of Random Forest Classifier, Logistic Regression and K Nearest Neighbours: 82.05% /
Accuracy of Support Vector Machine: 46.15%

### Conclusion

It can be devised from the results of the above model performance that Random Forest Classifier, Logistic Regression and K Nearest Neighbours have performed the best after applying under sampling technique and feature selection. Thus, the target variable of this classification problem can be predicted with utmost 82.05% accuracy.

### Reflection

This project aided in better understanding of various data analytics and visualization libraries used during Exploratory Data Analysis (EDA) before classification purposes. It also helped to study the unbalanced dataset and various methods to balance it such as eliminating null values, data standardization etc. This resulted in finding and combining different attributes/features as well as application of various machine learning algorithms to predict the target/output. 







