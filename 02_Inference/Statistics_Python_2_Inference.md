# Statistics with Python: Inferential Statistical Analysis with Python

My personal notes taken while following the Coursera Specialization ["Statistics with Python"](https://www.coursera.org/specializations/statistics-with-python), from the University of Michingan, hosted by Prof. Dr. Brenda Gunderson and colleagues.

The Specialization is divided in three courses and each one has a subfolder with the course notes.

1. [Understanding and Visualizing Data with Python](https://www.coursera.org/learn/understanding-visualization-data?specialization=statistics-with-python): `01_Visualization` 
2. [Inferential Statistical Analysis with Python](https://www.coursera.org/learn/inferential-statistical-analysis-python?specialization=statistics-with-python): `02_Inference`
3. [Fitting Statistical Models to Data with Python](https://www.coursera.org/learn/fitting-statistical-models-data-python?specialization=statistics-with-python): `03_Fitting_Models`

The present file is a guide or compilation of notes of the second course: **Inferential Statistical Analysis with Python**.

Mikel Sagardia, 2022.
No warranties.

Overview of contents:
1. Inference Procedures
   - 1.1 Example: Choosing Between Two Bags: A or B?
   - 1.2 Bayesian vs. Frequentist Statistics
   - 1.3 Statistical Notation
2. Python Lab: `./lab/01_PythonLab.ipynb` - Lists vs. Numpy Arrays, Dictionaries, (Lambda) Functions


## 1. Inference Procedures

Inference consists in:

1. Determining a parameter value with confidence (a mean, proportion, etc.), or
2. Testing theories about parameters: is the parameters higher than a value or not.

We usually have a **research question** we would like to answer which matches with one of those two approaches.

### 1.1 Example: Choosing Between Two Bags: A or B?

Example used in the course: we have two bags, A & B, which contain vouchers of chips of values `-1000, 10, 20, 30, 40, 50, 60, 1000`.

- A contains 20 chips amounting to a total sum of -560 USD
- B contains 20 chips amounting to a total sum of 1890 USD

The two different distributions of bags A & B chips are known. Our task is the following: we can draw only one chip and we need to decide which bag we'd like to keep; obviously, we want to predict which is bag B.

![Bag A and B Chip Distributions](./pics/bag_a_b_distributions.png)

By having a look at the distributions, we see that some decisions are easier than other:

- If we draw `-1000` or `1000`, we know clearly which bag is A or B.
- If we draw `60`, we know B is more likely than A.
- If we draw `30`, the decision is difficult, since likelihoods seem similar.

The example is used to introduce the following concepts:

- **Null hypothesis** and the **alternative hypothesis**: these are theories we want to test.
  - The null hypothesis is the hypothesis we are going to try to reject. In our case, we choose: "bag is A".
  - The alternate hypothesis is the hypothesis that is taken when the alternate is rejected; it is complementary to  the null. In our case: "bag is B".
  - Both hypothesis have a distribution!
- Decision rule: we overlap the distributions of both hypothesis and select a boundary to reject the null hypothesis; that boundary introduces two errors: one related to the null hypothesis (type I) and the other related to the alternate hypothesis (type II).
- Type I error: reject null when null is true: false positive (conservative).
- Type II error: we do not reject null when the alternative is true: false negative (catastrophic error).
- Significance level `alpha` = `p(Type I error)`; `alpha` is the probability of wrongly not rejecting the null hypothesis (Type I error); it is computed as the total probability accumulation beyond the decision boundary in the distribution of the null hypothesis (bag A).
- Power `beta` = `p(Type II error)`; `beta` is the probability of sticking to the null hypothesis when the alternate is true (Type II error); it is computed as the total probability accumulation bellow the decision boundary in the distribution of the alternate hypothesis (bag B).
- Note that `alpha` and `beta` are areas from different distributions but they share the same decision boundary (decision value).
- `p-value`: **probability of belonging to the null hypothesis distirbution**. We usually want to reject the null hypothesis, thus we want to have a small `p-value`. The threshold is often given by `alpha`, defined by the decision boundary. We know that if `p =< alpha`, the error of incorrect rejection is smaller than `alpha`. By convention, if `p > alpha`, we say we don't have enough evidence to reject the null hypothesis.

![Bag A and B Errors](./pics/bag_a_b_errors.png)

### 1.2 Bayesian vs. Frequentist Statistics

There are two frameworks or approaches in statistics:

- Frequentist: it measures events that have happened, thus the computations are unique an immutable; numbers are integers or boolean (correct or incorrect). Probabilities are made in the real world.
- Bayesian: computations are done considering that several scenarios could happen, and these are updated as we get more information; thus, numbers are more fractional. For instance, we select and answer and give to it a probability we think being correct. Probabilities are made in our minds.

These two different reasoning might lead to different probabilities. We need to understand them both.

### 1.3 Statistical Notation

- Mean of population vs sample: $\mu$ vs. $\overline{x}$ or $\hat{\mu}$
- Standard deviation: $\sigma$ vs. $s$ or $\hat{\sigma}$
- Proportions: $\pi$ or $p$ vs $\hat{\pi}$ or $\hat{p}$
- Confidence interval: the empirical rule states tha roughly the `2x sigma` spread covers the 95% of the total distribution; however, the exact multiplier is `1.96`.
- Prefer to use the term **standard error** to denote the true variability of a statistic, computed as its standard deviation.

See the colocated file `./Notation_Definitions.pdf`.

## 2. Python Lab: `./lab/01_PythonLab.ipynb` - Lists vs. Numpy Arrays, Dictionaries, (Lambda) Functions

This notebook summarizes some of the concepts contained in the python lab of the second* course. These are very basic things, although in the course they're introduced as intermediate.

It is supposed that these concepts build up on the concepts introduced in the previous first course.

Overview:

1. Lists vs. Numpy Arrays
2. Dictionaries
3. Functions
4. Lambda Functions
5. Reading Help Files
6. Assessment Code

## 3. Categorical Data: One Proportion

### 3.1 Estimating a Population Proportion with Confidence

When working with confidence intervals, our values are reported in a range:

`Best Estimate +- Margin of Error`

The `Best Estimate` refers to the population, but is actually computed from the sample: `p_hat`.

The `Margin of Error = MoE` is defined as "a few" estimated standard errors; if we have a **significance** of `0.05`, i.e., a **confidence interval** of `95 %` which would cover `95 %` of the possible values, 

`Margin of Error = 1.96 x SE`, with 

`SE = Standard Error = Standard Deviation of the Sample Proportion in the Sampling Distribution`. Note that this is not the standard deviation of the sample, but it is computed using it! The variance of the Bernoulli distribution is `p x (p-1)`. Thus the standard error is `SE = sqrt(Variance / n)`.

`Z*(95%) = 1.96` (that "a few" multiplier)

Example: a hospital polls toddler parents whether they use a car seat. The estimated parameter is the proportion of parents who use a car seat. Data:

- `n = 659` parents sampled.
- 540 responded 'yes'.

Proportion (sample): `p_hat = 540 / 659 = 0.82`.

**Standard error of a proportion**: `sigma = sqrt(p_hat x (1-p_hat)/n) = 0.014`

`95 % CI = 1.96 x SE = 0.029`

Solution (note formulation): Based on our sample of 659 parents with toddlers, with 95% confidence, we estimate that between `0.82 +- 0.03 = [0.79, 0.85]` of their total population uses car seats.

So, the sample is used to make an estimation of the population parameter! That's the magic of the confidence interval: while the best estimate refers to the sample, the region of confidence refers to the population.

### 3.2 Understanding Confidence Intervals

Confidence intervals are used to report **population** estimates based on computations performed with a **sample** measurements.

The `95% CI` is not `95%` the chance or probability of the population proportion being in that interval! Instead the `95% CI` relates to the level of confidence we have in the statistical procedure we used: if we draw samples and compute the CI with this procedure, the real parameter will be in the predicted range `95%` of the time!

![Confidence interval: interpretation](./pics/confidence_interval.png)

The more confident we want to be, the larger the multiplier is, increasing the range we report.

Some insights after playing with the [Chapter 4: Frequentist Inference - Section 2: Confidence Interval](https://seeing-theory.brown.edu/frequentist-inference/index.html#section2) from the [Seeing Theory](https://seeing-theory.brown.edu) website.

- A larger sample size reduces the value of the standard error (standard deviation of the sample), thus the range is decreased.
- A larger confidence interval `1 - alpha` requires a larger multiplier of the standard error; the increase is exponential. A larger multiplier leads to a larger range.

### 3.3 Assumptions for a Single Population Proportion Confidence Interval

We have the following assumptions:

- We have a **simple random sample (SRS)**: a representative subset of the population made by observations/subjects that have equal probability of being chosen. To check that, analyze how the data was collected and consider at least whether the sample is representative.
- We need to have a **large enough sample size**; that way, the distribution of sample proportions will tend to be normal. By convention, large enough is considered to be at least 10 observations of each class/category; example with car set usage: at least 10 "yes" and at least 10 "no".

### 3.4 Conservative Approach & Sample Size Consideration

If we are not sure if the sample is SRS we can take a larger or **conservative** standard error as if the estimated proportion were `p_hat = 0.5` (maximum standard deviation).

Then, with a `95% CI` (`MoE = 0.05`), we have:

`estimated p +- (1.96) * (0.5 / sqrt(n))`

which is approximately (cancelling 2):

`estimated p +- 1 / sqrt(n)`

That formula is very handy, because the range depends only on the sample size! To be more accurate, the **conservative margin of error** depends on

1. the sample size `n`
2. and the confidence interval (multiplier `Z*`) we choose.

We can further use that concept for computing the sample size required to have a margin or error of `0.03` (3%, since we are estimating proportions) in the proportion estimation with a confidence of `99%`:

`MoE = 0.03`
`Z*(99%) = 2.576`
`p_hat = 0.5`
`MoE = Z*(97%)/2 * 1 / sqrt(n)`
`n = ((Z*(97%) / 2) / MoE)^2 = 1843.27`
`-> n >= 1844`

## 4. Categorical Data: Two Proportions

Example: What is the difference in population proportions of parents reporting that their children age 6-18 have had swimming lessons, between white children and black children?

Pupulation: all parents with white children and all parents with black children.

Our parameter of interest is the different in population proportions: `p1 (white) - p2 (black)`; we want to compute the best estimate and its `95% CI`.

Collected data:
- Sample black: 247; 91 had swimming lessons.
- Sample white: 988; 543 had swimming lessons.

Formula:

`Best Estimate +- Margin of Error`

`Best Estimate = p_1_hat - p_2_hat`

`Margin of Error = Z*(95%) x SE(p_1_hat - p_2_hat)`

`Z*(95%) = 1.96`

`SE(p_1_hat - p_2_hat) = sqrt((p_1_hat x (1 - p_1_hat))/(n_1) + (p_2_hat x (1 - p_2_hat))/(n_2))`: The Standard Error of a difference of proportions is the sum of the variances square-rooted.

Result:

`p_1_hat = 0.55`

`p_2_hat = 0.37`

Thus: `0.18 +- 0.0677 = (0.1123, 0.2477)`.

### Interpretation & Assumptions

With 95% confidence, the population proportion of parents with white children who have taken swimming lessons is 11.23% to 24.77% higher than the population of parents with black children who have taken swimming lessons.

**If 0 is contained in the interval, we cannot say there are differences**; since in our case 0 is not in the interval, we can say that both proportions are different!

Assumptions:
- We have two independent random samples.
- We have large enough samples: we need to have at least 10 measurements for each of the 4 categories (black-yes, black-no, white-yes, white-no).

## 5. Quantitative Data: One Mean -- Estimating Population Mean with Confidence

Example (Cartwheel dataset): What is the **average** cartwheel distance (in inches) for adults? (distance from the forward foot before performing the cartwheel to the final foot after performing it).

Population: all adults.
Parameter of interest: population mean Cartwheel distance.
Sample size: 25.

We want to construct a `95% CI`.

Even though the data is not normally distributed (see QQ-plots and historgram in figure), we can still compute the `95% CI` if we fulfill our regular assumptions: (1) independent random collection of measurements, (2) large enough sample size.

![Cartwheel distance: diagrams](./pics/cartwheel_distance_diagrams.png)

The descriptive summary variables are used:

![Cartwheel distance: summary](./pics/cartwheel_distance_summary.png)

`Best estimate +- Margin of Error`

`Best estimate = Mean(Sample Measurements)`

`Margin of Error = T*(95%, n= 25) x Estimated SE`

`Estimated SE = Estimated Standard Error = sqrt(Estimated Variance / n) = StdDev(Sample Measurements) / sqrt(n)`. Note that the standard error is the spread of the sampling distribution, i.e., the error of the sample in the sampling distribution of samples. Our estimated value is computed by taking the measurements of our sample.

`T*(95%, n= 25)` is the multiplier, as before; in this case, instead of using the normal/standard distribution, we take the **Student's T Distribution**. This distribution depends on the sample size used. That variable is called **degree of freedom** (df).

`95% CI`

`n = 25 -> T*(95%, df = n=25) = 2.064`

`n = 1000 -> T*(95%, df = n=1000) = 1.962`

The T Distribution approximates to the normal distribution as the sample size increases.

Computing all terms:

![Cartwheel: confidence interval](./pics/cartwheel_distance_interval.png)

### Interpretation

With 95% confidence, the population mean cartwheel distance for all adults is estimated to be between 76.26 and 88.70 inches.

Recall the confidence refers to our procedure: if we repeat the measurements with different samples using the same methods, 95% of the intervals will contain the real mean!

## 6. Quantitative Data: Two Means -- Estimating a Mean Difference for Paired Data

**Paired data** arises when collected measurements are related, e.g.:

- Pre and post treatment measurements on the same subjects
- Measurements of twins
- Measurements of family members
- Measurements within the same lot/batch of production
- etc.

Example: What is the average difference between the older twin's and younger twin's self-reported education?

Population: all identical twins.
Parameter of interest: population mean difference of self-reported education.

We want to construct a `95% CI` fort he mean difference.

We have the following data:

![Twins: Education differences](./pics/twins_education_differences.png)

The formulas is the same as before, but we use the difference as the measurement, often notes with subscript `d`. Thus, the mean of the differences is taken and the standard deviation of the differences is used:

![Twins: Education differences CI](./pics/twins_education_differences_interval.png)

`Best Estimate +- Margin of Error`

`mean(differences) +- T*(95%, df = n) x (std(differences) / sqrt(n))`

Result:

`0.084 +- 0.0814 = (0.0025, 0.1652)`

### Interpretation & Assumptions

With 95% confidence, the population mean difference between the two paired groups is estimated to be between 0.0025 and 0.1652 years.

Since the reasonable range is on the positive side (i.e., 0 not contained), we conclude that the older twins have more education years on average. However, note that the interval almost contains 0. The key idea is that we need to check whether the 0 value is inside the range.

Assumptions:
- Random sample of identical twins.
- Population differences normal or large enough sample size.

## 7. Quantitative Data: Two Means -- Estimating a Mean Difference for Independent Groups

Now, we don't have paired data, but measurements of two unrelated groups.

Example: Do male and female BMI means differ significantly for the USA mexican-american adutls age 18-29? BMI = body mass index = kg/m^2.

Population: all male & female mexican-american adults age 18-29 in the USA.
Parameter of interest: difference of BMI means: `mu_1 - mu_2`.

Data summary:

![BMI means difference](./pics/bmi_means_difference.png)

In general, we have two approaches:

1. The pooled approach: the standard deviations of the two population groups are assumed to be equal.
2. The unpooled approach: the standard deviations of the two population groups are assumed to **not** be equal.

Formulas for the **unpooled approach**:

`Best estimate +- Margin of Error`

`Best estimate = Difference Means = mean(Group 1) - mean(Group 2)`

`Margin of Error = T*(95%, df = min(n_1-1, n_2-1)) x Estimated SE`

`Estimated SE = sqrt(std(Group 1)^2 / n_1 + std(Group 2)^2 / n_2)`

For the **pooled approach**, (1) the estimated standard error and (2) the df change:

![BMI means difference: CI interval computation for the pooled case](./pics/bmi_means_difference_result_pooled.png)

