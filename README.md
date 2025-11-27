 ADVANCED LINEAR REGRESSION — DETAILED NOTES

1. FEATURE ENGINEERING
Feature Engineering refers to the process of transforming raw data into meaningful features that improve model performance. In advanced linear regression, feature engineering is critical because linear models rely heavily on well-structured input variables.
1.1 Purpose of Feature Engineering
    • Improve model accuracy
    • Reduce bias and variance
    • Make relationships more linear
    • Handle non-linear effects
    • Manage noisy or missing data
    • Prepare data for interpretability

1.2 Types of Feature Engineering
(a) Handling Missing Data
    • Remove rows (only when few)
    • Impute:
        ◦ Mean/Median (numerical)
        ◦ Mode (categorical)
        ◦ KNN imputation (more advanced)

(b) Encoding Categorical Variables
Linear regression cannot handle strings → convert categories to numbers.
    • One-hot encoding
Creates binary columns for each category.
    • Ordinal encoding
Assigns numerical order (only when categories have ranking).
    • Binary encoding / Target encoding (advanced)

(c) Scaling Features
Important because:
    • Linear regression uses gradients
    • Large scales dominate coefficients
Methods:
       Standardization:
              xscaled=x−μσx_{scaled} = \frac{x - \mu}{\sigma}xscaled​=σx−μ​
    • Min-Max Scaling:
x′=x−min⁡max⁡−min⁡x' = \frac{x - \min}{\max - \min}x′=max−minx−min​
    • Robust Scaler (uses median, good for outliers)


(d) Feature Creation
Creating new variables from existing ones.
Examples:
    • Ratios (e.g., Price/Income)
    • Interaction terms (x1 × x2)
    • Polynomial features (x², x³, …)
    • Aggregations (e.g., monthly averages)
    • Time-based features (year, month, lag features)

(e) Handling Outliers
Methods:
    • Winsorization
    • Clipping extreme values
    • Log transformation
    • Box-Cox / Yeo-Johnson transformations

(f) Feature Selection
Reduces dimensionality.
Methods:
    1. Filter methods – correlation, chi-square
    2. Wrapper methods – forward/backward selection
    3. Embedded methods – Lasso/Ridge regression

(g) Transformations for Linearity
Linear regression assumes linear relationship between X and y.
Transformations include:
    • Log(x)
    • Sqrt(x)
    • 1/x
    • Power transformations
These help convert non-linear behaviour into linear form.

2. FEATURES AND POLYNOMIAL REGRESSION
Polynomial regression is an extension of linear regression that models non-linear relationships by adding polynomial terms.

2.1 Why Polynomial Regression?
Some features do not have a straight-line relationship with the target.
Example:
    • Price vs Age of a used car
– Price decreases rapidly first, then slowly
To capture curves → introduce polynomial features.

2.2 How Polynomial Regression Works
Given a feature ( x ), create polynomial terms:
[
y = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3 + \cdots + \beta_d x^d
]
This is still linear regression because:
    • It is linear in coefficients ( \beta )
    • Only the features are transformed

2.3 Polynomial Features
If ( x = [x_1, x_2] ) and degree = 2, polynomial expansion includes:
    • ( x_1 )
    • ( x_2 )
    • ( x_1^2 )
    • ( x_2^2 )
    • ( x_1 x_2 ) ← interaction feature
For degree = 3:
    • ( x_1^3 )
    • ( x_2^3 )
    • ( x_1^2x_2 )
    • ( x_1x_2^2 )

2.4 Interaction Features
These capture interactions between variables:
    • The effect of x1 depends on x2
Example:
    • Marketing × Holiday season
(Marketing is more effective during holiday seasons)

2.5 Choosing Polynomial Degree
    • Degree 1 → Linear regression
    • Degree 2 → Mild non-linearity
    • Degree 3 → Stronger curvature
    • Degree 4+ → Risk of overfitting
Use:
    • Cross-validation
    • Learning curves

2.6 Advantages
    • Easy to implement
    • Captures non-linear patterns
    • Still interpretable

2.7 Disadvantages
    • Prone to overfitting
    • High-degree polynomials oscillate wildly
    • More features → higher complexity

3. POLYNOMIAL REGRESSION IN SCIKIT-LEARN
Scikit-learn makes it easy to implement polynomial regression using:
    • PolynomialFeatures
    • LinearRegression
    • Pipeline

3.1 Import the Required Libraries
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

3.2 Simple Polynomial Regression Example
Step 1: Create a Pipeline
model = Pipeline([
    ('poly', PolynomialFeatures(degree=3)),
    ('lin', LinearRegression())
])
Step 2: Train (Fit)
model.fit(X, y)
Step 3: Predict
y_pred = model.predict(X)

3.3 Explanation of Code
PolynomialFeatures(degree=d)
    • Automatically adds:
        ◦ x
        ◦ x²
        ◦ x³
        ◦ … up to degree d
    • Adds interaction terms
    • Output feature space grows exponentially
Pipeline
    • Applies transformations in sequence
    • Ensures:
        ◦ No data leakage
        ◦ Clean and reproducible process

3.4 Getting Feature Names
Useful for interpretation.
poly = PolynomialFeatures(degree=3)
poly.fit(X)

poly.get_feature_names_out()

3.5 Scaling with Polynomial Regression
Polynomial expansion can create huge values → scale features.
Use:
from sklearn.preprocessing import StandardScaler

model = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=3)),
    ('lin', LinearRegression())
])

3.6 Polynomial Regression with Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('lin', LinearRegression())
])

model.fit(X_train, y_train)
preds = model.predict(X_test)

3.7 Model Evaluation
Use:
    • R² Score
    • MAE
    • RMSE
from sklearn.metrics import mean_squared_error, r2_score

r2 = r2_score(y_test, preds)
rmse = mean_squared_error(y_test, preds, squared=False)

3.8 Visualizing Polynomial Regression
For 1D x:
import matplotlib.pyplot as plt

plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.show()

SUMMARY TABLE
Topic	Key Points
Feature Engineering	Handling missing data, scaling, encoding, transformations, interactions
Polynomial Regression Concepts	Capture non-linearity by adding polynomial & interaction terms
Polynomial Regression in Scikit-learn	Use PolynomialFeatures + LinearRegression + Pipeline

