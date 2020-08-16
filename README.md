# US-House-Price-Prediction
 Multi ML Project, Regression Models vs Classification Models

un bash shell script 'run.sh' which finds & installs packages listed in requirements.txt
Libraries: jupyter 3.6.7 matplotlib numpy pandas scikit-learn

dir tree
src ├── module1.py ├── module2.py ├── eda.ipynb ├── run.sh ├── requirements.txt ├── README.md

Overview Explanation
The Jupyter Notebook contains the Exploratory Data Analysis done on the 'sales' dataset, preprocessed with ms excel saved as home_sales-formetdate.csv for further exploratory on jupyter notebook.

The 'module1.py' file containts 3 regression models: - 1: Linear Regression model - 2: Lasso Regression model - 3: Ridge Regression model

The 'module2.py' file containts 3 classification models: - 1: Random Forest model - 2: Xgboost model - 3: ANN - Artificial Neural Network model

Execute the 'run.sh' script on bash and run 'module1.py' and 'module2.py' file.


=======
Approach 1 Regreesion Model vs Approach 2 Classification Model:
A classification algorithm may predict a continuous value, but the continuous value is in the form of a probability for a class label. A regression algorithm may predict a discrete value, but the discrete value in the form of an integer quantity.
Some algorithms can be used for both classification and regression with small modifications, such as decision trees and artificial neural networks. Some algorithms cannot, or cannot easily be used for both problem types, such as linear regression for regression predictive modeling and logistic regression for classification predictive modeling.

Importantly, the way that we evaluate classification and regression predictions varies and does not overlap, for example:
Classification predictions can be evaluated using accuracy, whereas regression predictions cannot.
Regression predictions can be evaluated using root mean squared error, whereas classification predictions cannot.

The Regression Models "appeared" to be having higher accuracy compared to Classification Model (without binning), on preliminary observation it could be overfitting issue on the Regression Models, needed more time to exlpore further (data augmenting or further features engineering needed). 

========
EDA
I have scripted a "SQLite3 convert .db file to .csv file" before carrying out a visual inspection on the datasets via excel for those easy & immediate rectifying; spell check, formating etc (5%), which the rest of the preprocessing is done via python Jupyter Notebook (95%)

On EDA findings, I would suggest:
(1) as majority of hse sales is between shaded area (Lat 47.2 to 47.8, Long -121.8 to -122.4), within the shaded area high sales between area (Lat 47.5 to 47.8, Long -122 to -122.4), real estate company can advertise more in the areas to be more effective in the marketing.

(2) as the 2 or more bathrm hse's min price is higher, esp 4 bathrm min is abt $0.25M, shd focus on 2 more bathrms hses

(3) as 1 to 2 flr hse's min price is almost same, 2.5 to 3had higher min price, 3.5 had higher price but lesser transactions, shd forget about 1 to 2 flr hses, focus on 2.5 to 3 flr hses

(4) as most sales happened for hse with basement below 1000sft, above 1000sft command higher min price, effort shd be on basement below 1000sft

(5) most optimal sales shd be hse with living rm btw 2000 to 4000 sft as transactions is high & price is higher too

(6) Lot-size btw 10000 to 15000sft had higher min price & high transactions vol

(7) Hse Condition doesn't seem to give better min price

(8) Hse age, seem the older hse is not affected, thus agent can tgt and reno such old hse to sell

The above 8 Key Points shd constitute a basis for the real estate company to furnish information for their clients, and also formulate strategies on marketing & expound higher margin hse transactions.
