from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_absolute_error,mean_squared_error
from sklearn.datasets import load_iris

iris = load_iris() #['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)'],


x = iris.data         # iris['data']
y = iris.target       # iris['target']

print(iris['target_names'])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


Lin = LogisticRegression(random_state=0)

Lin.fit(x_train,y_train)

Pred_y = Lin.predict(x_test)

acc = accuracy_score(y_test,Pred_y)
print(acc)

#MAE measures the average magnitude of the errors in a set of predictions,without considering their direction.
#The Mean Absolute Error(MAE) is the average of all absolute errors.
print(mean_absolute_error(y_test,Pred_y))


#RMSE is a quadratic scoring rule that also measures the average magnitude of the error.
# The average squared difference between the estimated values and the actual value
print(mean_squared_error(y_test,Pred_y))