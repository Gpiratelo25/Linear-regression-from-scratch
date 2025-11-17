# Linear Regression — Gradient Descent from Scratch


## Objective
* The goal of this project is to deepen my understanding of the concepts inside the linear regression implementing Gradient descendent from scracth and then compare to scikit learn.

## Dataset
* The dataset used was the `fetch_california_housing`
* Number of instances: 20640
* Attribute Information:
    - MedInc        median income in block group
    - HouseAge      median house age in block group
    - AveRooms      average number of rooms per household
    - AveBedrms     average number of bedrooms per household
    - Population    block group population
    - AveOccup      average number of household members
    - Latitude      block group latitude
    - Longitude     block group longitude
* The target variable is the **median house value** in California Districts expressed in **hundreds of thousands of Dollars**
* This is a good dataset to implement the linear regression, the values are transformed to numbers(the way the model understand) and the target variable are correlated with the others variables
* I manually implemented the formulas to make the standard scaler, this is a very important step to put all the variables at the same scale:
```python
        X_mean = X_train.mean(axis=0)
        X_std  = X_train.std(axis=0)

        X_train_s = (X_train - X_mean) / X_std
        X_test_s  = (X_test - X_mean) / X_std
```





## Method

* To upload the data used 
```python
    dados = fetch_california_housing()
    X = dados.data
    y = dados.target
```
* Then split
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```
* We used the standard scale to put for example, the house age ( in years) and the AveRooms (quantity) at the same scale to make easier for the model to learn the relationship
* To implement the Gradient Descent we first create a matrix of weights of zeros and the intercept as zero
    - We first use this formula, the prediction:
    `ŷ = Wx + b`
     Where W is the weight, x the values and b the intercept
    - After we calculate the residual values 
        `residuo=yhat-y_train`
    
   then the loss (MSE):
    `loss=(residuo**2).mean()`
    -**Gradient for the Weights** -> Here the important step is where we reorganize the x_train transposing it and multiplying by the feature-residual interaction, with this, we show how each variable has on the dataset, then multiply by the 2/len(x_train)
        ```        dW=(2/len(X_train_s))*X_train_s.T.dot(residuo)```
    - Then calculate the intercept and refresh the variables multiplying by the learning rate:
        ```
            #Aqui calculamos o intercepto
            db=(2/len(X_train_s))*residuo.sum()
            #atualizamos os pesos
            W-=i*dW
            b-=i*db
        ```
    - We create a visualization to see the curves of the losses:
    <img width="554" height="432" alt="image" src="https://github.com/user-attachments/assets/6fac673a-d18e-4661-b361-367464f3465e" />

    - The Result:
    ## Hyperparameter Tuning — Learning Rate (Gradient Descent)

| learning_rate | epochs | final_loss | R² | MSE | RMSE |
|--------------:|--------:|------------:|------:|-----------:|-----------:|
| **0.0001**    | 7549    | 0.877738    | 0.341369 | 0.864479 | 0.929774 |
| **0.001**     | 1411    | 0.632167    | 0.527036 | 0.620784 | 0.787898 |
| **0.01**      | 472     | 0.555822    | 0.580604 | 0.550474 | 0.741939 |
| **0.1**       | 123     | 0.526702    | 0.595670 | 0.530699 | 0.728491 |
| **0.5**       | 25      | 0.534184    | **0.597438** | **0.528379** | **0.726897** |

### Interpretation

The learning rate strongly influenced both convergence speed and model performance.

- Very small LR (0.0001) converged extremely slowly (7549 epochs) and produced weaker performance (R² = 0.34).
- Moderate LR values (0.001 and 0.01) provided a stable and fast convergence with good generalization.
- LR = 0.1 converged in only 123 epochs and achieved strong performance (R² = 0.595).
- The most aggressive LR (0.5) converged in only 25 epochs and reached the best R² (0.597).  
  However, this value is close to the stability limit and may oscillate in other datasets.

Overall, LR = 0.01 and LR = 0.1 represent the best trade-off between speed, stability, and accuracy.

## Comparison with Sklearn
* At the end we create the same model using the sklean.LinearRegression()
```
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LinearRegression())
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)
print(f"1D | MSE={mse:.5f} | R2={r2:.5f} | RMSE={rmse:.5f}")
```
-The result : `1D | MSE=0.53057 | R2=0.59577 | RMSE=0.72840`

* And the graph with the prediction curve :
    <img width="562" height="434" alt="image" src="https://github.com/user-attachments/assets/0bb53ecb-607f-4071-b9aa-54ab3949acfc" />


## Conclusion

This project taught me how Linear Regression really works behind the scenes and how strongly training depends on hyperparameters such as the learning rate. I clearly saw how a small change in the learning rate drastically affects the shape of the loss curve and the number of epochs required for convergence.

I didn’t expect such a big difference in training time and stability when switching between learning rates. The experiments showed how important it is to analyze the trade-off between speed and stability, and how Gradient Descent can behave very differently depending on this choice.

Overall, this project helped me build a deeper understanding of the optimization process, and it reinforced how important it is to tune hyperparameters carefully when training machine learning models.

