In order to fit data into a curve then combine them into a plane or 3D plot, we use the following six regression methods in machine learning and calculate fitting score to evaluate various methods to find the best one for each query and situation.
LinearRegressionisa linear approach for modeling relationship between a scalar dependent variable y and one or more explanatory variables (or independent variables) denoted X, which is under fitting in our research.
SVR(Support Vector Regression) [11]is a method of Support Vector Classification, whose basic principle is to find a regression plane which makes theminimum distance for each one in the data set to the plane.
This model only depends on a subset of training data, because the cost function for building the model doesn¡¯t care about training points that lie beyond the margin.
Nearest Neighbors Regressionis mostly used in cases where the data labels are continuous rather than discrete variables. The label assigned to a query point is computed based the mean of the labels of its nearest neighbors.
KNeighborsRegression is one of the Nearest Neighbors Regression which implements learning based on the k nearest neighbors of each query point, where k is an integer value specified by the user.
DecisionTreeRegressionis a basic classification and regression method. Three main steps are as follows:
1.Feature selection: using information gain criterion algorithm select feature£»
2.Decision tree generation: usingclassical ID3 algorithm to generate tree;
3. The pruning of the decision tree: in order to prevent over fitting phenomenon.
The generation of the decision graph corresponds to the local selection of the model, while the pruning of the decision tree takes into account the global minimum selection.
RandomForestRegressionis the expandingof the decision tree regression whichestablishes a forest with many decision trees in it in random way. There is no correlation between every decision tree in the random forest. The predicted value is the weighted average value of the target variable of the leaf node.
Gradient Boosting Regressionis the expandingof the decision tree regression.
Compared with the traditional gradient learning, Gradient Boosting Regression chooses the direction of gradient descent to ensure that the final results are best at the time of iteration.
Method Evaluation:
To evaluate these methods, we calculate the fitting score for each of them. Estimator in sklearn has a score method, which provides a default evaluation rule to solve the problem. Each score calculate the residual sum of squares ((y_true - y_pred) ** 2).sum() and the total sum of squares ((y_true - y_true.mean()) ** 2).sum().
Through these fitting scores we could figure out the best method through comparison. Under fitting means that the complexity of the model is too low to fit all the data well, and the training error is large, while over fitting shows that the model complexity is too high, the training data are too little, the training error is small, and the test error is big.
