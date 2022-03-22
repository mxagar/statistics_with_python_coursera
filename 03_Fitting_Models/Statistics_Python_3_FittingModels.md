# Statistics with Python: Inferential Statistical Analysis with Python

My personal notes taken while following the Coursera Specialization ["Statistics with Python"](https://www.coursera.org/specializations/statistics-with-python), from the University of Michingan, hosted by Prof. Dr. Brenda Gunderson and colleagues.

The Specialization is divided in three courses and each one has a subfolder with the course notes.

1. [Understanding and Visualizing Data with Python](https://www.coursera.org/learn/understanding-visualization-data?specialization=statistics-with-python): `01_Visualization` 
2. [Inferential Statistical Analysis with Python](https://www.coursera.org/learn/inferential-statistical-analysis-python?specialization=statistics-with-python): `02_Inference`
3. [Fitting Statistical Models to Data with Python](https://www.coursera.org/learn/fitting-statistical-models-data-python?specialization=statistics-with-python): `03_Fitting_Models`

The present file is a guide or compilation of notes of the second course: **Fitting Statistical Models to Data with Python**.

Mikel Sagardia, 2022.
No warranties.

Overview of contents:
1. Considerations for Statistical Modeling

## 1. Considerations for Statistical Modeling

### 1.1 Fitting Models to Data

We fit models to data, not the other way around!

We want to fit models to the data to

- estimate distribution properties of variables
- summarize relationships between variables
- predict values of variables.

We are going to use parametric models: the model is expressed in parameters; we find the parameter values that best fit the data. We are basically making inferences of parameters; thus, we can also compute their confidence intervals or we can perform hypothesis tests on them.

It is fundamental to assess how well the model fits the data; we're going to see techniques for that.

Example: test performance of students vs. student age; can age predict the test performance? Two major hypothesis are analyzed, which are subclasses of polynomial regression:

1. Mean-only model: `performance = mean + error`
2. Curvi-linear model: `performance = a + b*age + c*age^2 + error`

In both cases, `error = N(0, sigma^2)`. Thus, we estimate

1. 2 parameters for the first model: `mean`, `sigma^2`,
2. 4 parameters for the second model: `a`, `b`, `c`, `sigma^2`.

Note that:
- We can compute the standard error of the parameters (except `sigma`)
- We need to check the assumption `error = N(0, sigma^2)`; for that: we plot for the residuals
  - scatterplot
  - histogram
  - QQ-plot

If the errors are clearly not normally distributed, the model does not represent the data.

![Fitting Model: Performance vs. Age - Mean](./pics/fitting_models_example_mean.png)

![Fitting Model: Performance vs. Age - Curvilinear](./pics/fitting_models_example_curvilinear.png)

Personal notes:

- In both models, we are predicting the mean performance; however, the second model has polynomial terms of the variable age of higher order, whereas the first model does not have a variable of dependence: simply a mean is computed.
- The `error` term is `error = real performance - prediction`.


### 1.2 Types of Variables in Statistical Modeling

Roughly, we have these types of variables

- Categorical vs. Continuous: that relates to the type.
- Dependent Variables and Independent Variables: we want to find their relationship.

**Independent Variables** (IV; aka. predictors, regressors): these are the variables we want to predict.

- They can be manipulated, i.e., groups randomly assigned in the study, or observed.
- If observed, we can find relationships
- If manipulated/controlled, we can predict and in some situations make causal inferences.
- Can be categorical, continuos; if categorical (e.g., ethnicity), we cannot make functional relationships, instead we compare groups.

**Dependent Variables** (DV; aka. outcome, response, variables of interest):

- We select a reasonable distribution (e.g., normal) or the predicting variables and define its parameters (e.g., mean) as a function of the IVs (independent variables)
- Can be continuous, categorical

**Control variables**: control variables are independent variables added to the model in cases where we know they have a relationship with another independent variable used as a predictor. For instance: predict blood pressure as a a function of gender; since weight means are expected to be different for both genders and weight might be related to blood pressure, we add it to the model. That way, we can adjust for confounding.

**Missing data**: before any model fitting, we need to check for missing data:

- Perform bi-variate analyses on IVs and DVs
- We say a unit is a data-point with all independent variables or measurements and its associated dependent variable(s).
- By default, if a unit of analysis has a missing dependent variable, the entire unit is dropped; thus, we introduce bias, which will be larger the more different the dropped unit is
- Therefore, we should analyze how different the dropped units are.
- If there are differences, a possible approach is to predict the missing data with **imputation**.

### 1.3 Different Study Design Generate Different Data

We need to be well informed before staring to fit models.

Simple Random Samples (SRS) typically produce independent and identically distributed data (i.i.d.); thus, we can use the assumption that the observations are independent from each other. Thus, the standard error is going to be smaller, i.e., we're going to have more precise estimates.

Clustered Samples are related to measurements in randomly selected groups; it is to be expected that each group will have more similar data. Thus, there is going to be a correlation, i.e., they are not independent from each other, and we need to take that into account! In fact, we're going to have larger standard errors, i.e., the estimates are going to be less accurate.

In longitudinal studies, repeated measures of the same variable collected from the same unit (e.g., subject) are done over time; thus, they are expected to be correlated. The recorded variables are not independent from each other!

### 1.4 Objectives of Model Fitting: Inference vs. Prediction

When fitting a model, we have two major objectives:

1. We want to make and inference about relationships: which is the relationship?
2. We want to predict/forecast future outcomes using historical data.

Let's analyze each objective in the model:

`performance = a + b*age + c*age^2 + error`

**Objective 1**: Making Inferences

We compute the standard error of each and the `T-Statistic` using the `H0` that `a = b = c = 0`, i.e., there is no model. For instance, for the parameter `a`:

`Ta = (a - 0) / SE(a) = (5.11 - 0) / 0.10 = 51.1` -> `p-value = 0` -> `H0` is rejected, `a` is significant!

![Inferences with a Fitted Model](./pics/fitting_models_example_inference_parameters.png)

Now, each coefficient has a meaning:

- `a` represents the mean test performance when the age is equal to the mean of the dataset.
- `b` represents the rate of increase in performance with age.
- `c` represents the acceleration of increase with age.

If any of the parameters is non-significant, it drops from the model; and I understand we need to re-compute/fit it?

**Objective 2**: Making Predictions

To predict, we apply the model formula to an independent set of variables (in this case, `age`) and we get the estimate mean of the dependent variable (`performance`). We can also predict something different to a mean, such as a percentile.

However, we need to account for the `error`, or the uncertainty associated to the prediction: we need to report it, too, since it's part of the prediction model! I understand that we would report something similar to a confidence interval?

### 1.5 Plotting Predictions and Prediction Uncertainty

When we fit a model we get its parameter values; as important as the parameter values is their uncertainty, which is given by the **standard error** of the computed parameters. With that standard error, we can test any `H0` hypothesis with a `T-Test`:

`T = (Parameter Value - H0) / SE(Parameter Value)`

`H0` is selected usually to be `0`, but we can take any value! Then we would reject or not that value.

Note that the same model with the same parameter values can have a very different underlying dataset. An example is given for two datasets in which the same model is fit `Y = a + b*X`. Additionally, the parameters `a, b` have the same values:

![Model Uncertainty: Standard Error of Parameters](./pics/uncertainty_standard_error.png)

However, the points of the second dataset are much more spread away from the model!

Lessons:

- We need to always plot the dataset and the model.
- We need to compute the standard error of each parameter, as well as its `p-value`.
- The standard error needs to be plotted too: that's the gray area around the model.
- Large standard errors denote cases in which our estimates deviate too much from the data points.