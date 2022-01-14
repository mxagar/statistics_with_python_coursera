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