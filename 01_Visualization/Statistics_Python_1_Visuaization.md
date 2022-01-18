# Statistics with Python: Understanding and Visualizing Data with Python

My personal notes taken while following the Coursera Specialization ["Statistics with Python"](https://www.coursera.org/specializations/statistics-with-python), from the University of Michingan, hosted by Prof. Dr. Brenda Gunderson.

The Specialization is divided in three courses and each of one has a subfolder with the course notes.

1. [Understanding and Visualizing Data with Python](https://www.coursera.org/learn/understanding-visualization-data?specialization=statistics-with-python): `01_Visualization` 
2. [Inferential Statistical Analysis with Python](https://www.coursera.org/learn/inferential-statistical-analysis-python?specialization=statistics-with-python): `02_Inference`
3. [Fitting Statistical Models to Data with Python](https://www.coursera.org/learn/fitting-statistical-models-data-python?specialization=statistics-with-python): `03_Fitting_Models`

The present file is a guide or compilation of notes of the first course: **Understanding and Visualizing Data with Python**.

Mikel Sagardia, 2022.
No warranties.

Overview:
1. What is Statistics?
2. What is Data?
3. Python Intro + Jupyter Notebooks
4. What Can You Do with Univariate Data?
5. Python for Univariate Data Analysis

## 1. What is Statistics?

Statistics is a set of methods that can be applied to learn insights from data.

A "statistic" is a numerical of graphical summary of the data: average, percentage, minimum, etc.

The field of statistics is evolving and new toold are appearing able to tackle problems impossible until recently; for instance: Big Data.

Note that data can be missleading; statistics has tools to avoid missunderstandings and inform for a better decision-making. For instance: data summarization for better uderstanding. Note that we often talk about central values (e.g., averages) when we summarize, but the variation is as important as the central value.

The different perspectives of statistics:
- summarizing
- science ofuncertainty
- science of decisions
- science of variation
- art of forecasting
- science of measurement
- basis for principled data collection

Probability pre-dates statistics: it deals with probability distributions, randomness and uncertainty.
Data science uses statistics, but introduces new technical disciplines, such as databases, machine learning techniques, etc.

## 2. What is Data?

Data can be represneted in numbers, images, words, audio.

Some cool visualizations from [https://flowingdata.com](https://flowingdata.com):

- [A Day in the Life of Americans](https://flowingdata.com/2015/12/15/a-day-in-the-life-of-americans/)
- [A Day in the Life: Work and Home](https://flowingdata.com/2017/05/17/american-workday/)
- [Most Common Use of Time, By Age and Sex](https://flowingdata.com/2015/11/30/most-common-use-of-time-by-age-and-sex/)

### 2.1 Where Do Data Come From?

We distinguish two types of data:

1. Organic or process data: we collect data of phenomena that have occurred naturally: weather, etc.
   - Big Data uses these types of datasets, which can become enormous
   - In the case of Big Data, storing and accessing those data is a challenge and a research topic
2. "Designed" data collection: we collect data of experiments or samplings we have created
   - Individuals are sampled from the complete population and some data is collected
   - Usually, these datasets are smaller and much easier to work with, at least computationally

A general question for datasets is wether they are **i.i.d.** = **independent and identically distributed**. It is the case if

- Each observation of a sample is independent from all the others
- The values that we are observing arise from some common (same?) distribution

Having i.i.d. datasets allows us to make some assumptions and use some statistical tools. However, data is not always like that; examples in test scores:

- Students who sit next to each other tend to have similar scores
- Males and females might have different means
- Students from the same discussion section may have similar scores

In the case of not i.i.d data, we need to apply other statistical procedures.

We need to consider where data comes from in order to decide which assumptions and tools to apply.

### 2.2 Variable Types

- Quantitative variables:
  - continuous (height; also age is considered continuous)
  - discrete (members in a household).
- Categorical variables: 
  - ordinal: groups have ranking (role ranking: junior, senior, etc.)
  - nominal: groups have merely names (race)

### 2.3 Study Design

There are many types of study designs and sometimes it is hard to classify one study to belong to a type.

Some notions:
- **Exploratory vs Confirmatory** studies
  - Exploratory studies collect data without first pre-specifying questions
  - Confirmatory studies suggest a falsifiable hypothesis and they test it; so they collect data for that. That is more related to the scientific method
- **Comparative vs Non-Comparative** studies
  - We often compare different treatments of a thing: a medicine, etc.
  - But there are other studies that simply try to predict something without any comparison
- **Observational vs Experimental**
  - Observational studies treat the subjects in the same group and insights arise from observation; it is said that the subjects are exposed to conditions, whereas in experimental studies they are said to be assigned
  - Experimental studies are more structured and divide often subjects into groups for targeted experiments

#### Power and Bias

**Power analysis**: process to assess whether a given study design will lead to meaningful results.

**Bias**: Measurements that are systematically off-target or sample is not representative of population of interest. Observational studies are specially vulnerable to it.

### 2.4 Data Management and Manipulation

The general shape and properties of a dataframe table is explained in a text: subject x variables.
Nothing special to be noted here, except some terminology:
In *repeated measures* tests, data can be stored in *wide* or *long* format.
Repeated measures studies measure dependent variables of the same subject under different treatments or conditions:
- wide format refers to teh fact that each subject has a row with more columns, related to the outcomes of different treatments
- long format refers to the fact that each subject-treatment is a row

## 3. Python Intro + Jupyter Notebooks

Very few comments made here, since I have year-long experience with python and Jupyter.

Interesting link on Markdown: [Markdown reference/cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet).

We can create our own notebooks in the course environment/workspace.

After a brief python introduction, `pandas` basics are shown and the two main datasets used throughout the course are tried. The following notebook summarizes it all:

`./lab/01_DataManagement.ipynb`.

The contents in the notebook:

1. Pandas DataFrames: Intro with the Cartwheel Dataset
   - Selection
   - Group By
2. Pandas DataFrame - NHANES Dataset
3. Python guidelines
   - Cheatseets: NUmpy, Pandas, Scipy, Matplotlib
   - Style guidelines (based on Google)

## 4. What Can You Do with Univariate Data?

### 4.1 Categorical Data: Tables, Bar Charts & Pie Charts

Categorical data classifies the samples into groups; e.g.: marital status in the NHANES dataset.
We can summarize this data with **frequency tables**: count & percept per group/class.
We can visualize these data with **bar charts** or **pie chart**.
- Pie charts are not recommended, since not so intuitive and small parts hard to see
- We can order nominal data from larger to smaller

### 4.2 Quantitative Data: Histograms

Whenever we can quantitative data (discrete: age, continuous: height), the **histogram** is a perfect tool to analyze the distribution of the data and get a first idea of what is happening. Four main aspects observed:
- **Shape**: symmetric, left/right skewed (skew side is where the tail is), bell-shaped, unimodal (one peak = mode) / bimodal, etc.
- **Center**: mean & median (1/2 area); note that the mean is on the side of the tail (it is sensitive to the extreme observations) and the median on the side of the peak (mode). In general, the median is considered a robust center estimation, in contrast to the mean.
- **Spread**: Range (max - min), Inter-Quartile Range, Standard deviation & variance. A standard deviation S means that on average samples are the about S points away from the mean (important insight).
- **Outliers**: data points far from the bulk. With skewed distributions they appear usually. If considerable outliers, we can also ay where most of the area/mass is.
s
These four aspects are summarized in 1-2 sentences if the data distribution is described in text.

### 4.3 Quantitative Data: Numerical Summaries = Quartiles

Five summary values are given for a distribution, which are the **min and max values** and **the quartiles 25% (Q1), 50% (Q2 = median), 75% (Q3)**. 

The IQR is: `IQR = Q3(75%)-Q1(25%)`. Typically, when the data is skewed, we provide the IQR, since it describes better where most of the data is.

### 4.4 Empirical Rule & Standard Score

In a bell-shaped curve we often have a normal distribution $N(\mu,\sigma)$.
The **empirical rule of 68-95-99.7** tells us the percentage of samples that lie in ranges:
- 68% of samples are 1 standard deviation away from the mean: $\mu \pm \sigma$; note that the range is $2\sigma$.
- 95% of samples are 2 standard deviations away from the mean: $\mu \pm 2\sigma$
- 99.7% of samples are 3 standard deviations away from the mean: $\mu \pm 3\sigma$

The **standard score** is also known as the **Z-score** for the normal distribution; it tells us how far away a measurement/sample is from the mean and on which side:

$Z = \frac{x - \mu}{\sigma}$

Note that SS tells precisely how many $\sigma$ units our sample is from the mean. Additionally, recall the empirical rule in that sense.

Thus:
- If SS is close to 0, we are close to the mean
- If SS < 0, we are below the mean, if SS > 0, above
- The bigger the absolute of SS, the further away from the mean

![The empricial rule and the standard score](./pics/empirical_rule.png)

### 4.5 Quantitative Data: Box Plots

Any distribution of quantitative data can be visualized with a box plot.
A box plot is a graphical representation of the quartiles or the five-value summary: min, max, 25%, median, 50%, IQR.
Additionally, if the whiskers are not set in the min/max locations, outliers can be detected.
There is a technique for that.

Most interesting box plots appear when we plot different groups together, i.e., when we facet the same quantitative variable of different qualitative variables (groups). For example:
- Blood pressure by age ranges and gender.
- Iris dataset: sepal length by species.

![The box plot](./pics/box_plot.png)

Nice interactive plotting example: [https://markkurzejaumich.shinyapps.io/bivariate_analysis/](https://markkurzejaumich.shinyapps.io/bivariate_analysis/).

### 4.6 Commenting Graphs

Quiz/assignment in which several graphs were shown and they needed to be described, highlighting the insights in them.

### 4.7 Links and Modern Infographics

Interesting links from [The New York Times: What's Going on in This Graph?](https://www.nytimes.com/column/whats-going-on-in-this-graph), which encourages discussions on graphs:
- [Climate](https://www.nytimes.com/interactive/2018/05/03/learning/08WGOITGraphLN.html)
- [Nutrition](https://www.nytimes.com/2017/10/09/learning/whats-going-on-in-this-graph-oct-10-2017.html)
- [Essay publishing](https://www.nytimes.com/interactive/2018/02/08/learning/13WGOITGraphLN.html)

Infographics are increasing in popularity; a few caveats or points to assess their quality:
- Look at the scale of each dimension
- If areas/volumes are shown, be aware of missinterpretation due to the fact that areas/volumes grow quadratically/cubically in terms of the size variable

Some interesting links:
- [History of Infographics](https://www.smithsonianmag.com/history/surprising-history-infographic-180959563/)
- [How to Read Infograhics](https://www.themarginalian.org/2013/10/08/best-american-infographics-david-byrne/)

## 5. Python for Univariate Data Analysis

