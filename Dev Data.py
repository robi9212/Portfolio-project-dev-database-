import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split

dev_data = pd.read_csv('developer_dataset.csv', low_memory=False)

# - reviewing columns of dev_data
print(dev_data.columns)

# - identifying columns with missing data
print(dev_data.count())

# - This will only work for numerical columns, but that will still be helpful.
# - average values
# - max and min values
# - the number of missing data points
print(dev_data.describe())

# - percentage missing data for each column
maxrow = dev_data.RespondentID.count()
print('% Missing Data:')
print((1 - dev_data.count()/maxrow) * 100)

# - dropping columns with largest % of missing data
dev_data.drop(['NEWJobHunt', 'NEWJobHuntResearch', 'NEWLearn'], axis=1, inplace=True)
print(dev_data.columns)

# - Determine what kind of missing data you have for employment and developer type, 
# - at a country level, where the data is missing for each field:
dev_data[['RespondentID', 'Country']].groupby('Country').count()
missing_data = dev_data[['Employment', 'DevType']].isnull().groupby(dev_data['Country']).sum().reset_index()

# - Create a bar plot for the number of missing values in the 'Employment' column
A = sns.catplot(
     data=missing_data, kind="bar",
     x="Country", y="Employment",
     height=6, aspect=2)

# - Create a bar plot for the number of missing values in the 'DevType' column
B = sns.catplot(
    data=missing_data, kind="bar",
    x="Country", y="DevType",
    height=6, aspect=2)
plt.show()

# - removing columns with missing data from employment and devtype
dev_data.dropna(subset=['Employment', 'DevType'], inplace=True, how='any')

# - aggregate the employment data by key developer roles that align with major parts of the development lifecycle:

# - Front-end
# - Back-end
# - Full-stack
# - Mobile development
# - Administration roles
empfig = sns.catplot(x='Country', col='Employment', data=dev_data,
                    kind='count', height=6, aspect=1.5)
# - Focus on a few of the key developer types outlined in the Stack Overflow survey
dev = dev_data[['Country','DevType']].copy()
dev['BackEnd'] = dev['DevType'].str.contains('back-end')
dev['FrontEnd'] = dev['DevType'].str.contains('front-end')
dev['FullStack'] = dev['DevType'].str.contains('full-stack')
dev['Mobile'] = dev['DevType'].str.contains('mobile')
dev['Admin'] = dev['DevType'].str.contains('administrator')

dev = dev.melt(id_vars=['Country'], value_vars=['BackEnd', 'FrontEnd',
                                             'FullStack', 'Mobile', 'Admin'],
                                             var_name='DevCat', value_name='DevFlag')
dev.dropna(how='any', inplace=True)
devFig = sns.catplot(x='Country', col='DevCat', data=dev,
                     kind='count', height=6, aspect=1.5);
devFig.set_xticklabels(rotation=45, ha='right')
plt.show()

# - taking a look at the distribution of majors over each year
missing_undergrad = dev_data['UndergradMajor'].isnull().groupby(dev_data['Year']).sum().reset_index().copy()
sns.catplot(x='Year', y='UndergradMajor', data=missing_undergrad,
            kind='bar', height=4, aspect=1)

# - carry that value backwards for each participant to fill in any missing data. 
# - This is a great use for one of our Single Imputation techniques: NOCB! Fill in the gaps using NOCB:
dev_data = dev_data.sort_values(['RespondentID', 'Year'])
dev_data['UndergradMajor'].bfill(axis=0, inplace=True)

# - analyze the major distribution for each year, using a vertical bar chart visualization:
# - Key major groups outlined in the Stack Overflow survey
# - Key major groups outlined in the Stack Overflow survey
majors = ['social science','natural science','computer science','development','another engineering','never declared']
education = dev_data[['Year','UndergradMajor']]

# need to fix
#############################
# education.loc[education['UndergradMajor'].str.contains('(?i)social science'), 'SocialScience'] = True
# education.loc[education['UndergradMajor'].str.contains('(?i)natural science'), 'NaturalScience'] = True
# education.loc[education['UndergradMajor'].str.contains('(?i)computer science'), 'ComSci'] = True
# education.loc[education['UndergradMajor'].str.contains('(?i)development'), 'ComSci'] = True
# education.loc[education['UndergradMajor'].str.contains('(?i)another engineering'), 'OtherEng'] = True
# education.loc[education['UndergradMajor'].str.contains('(?i)never declared'), 'NoMajor'] = True
#############################

education = education.assign(
    SocialScience=education['UndergradMajor'].str.contains('social science', case=False),
    NaturalScience=education['UndergradMajor'].str.contains('natural science', case=False),
    ComputerScience=education['UndergradMajor'].str.contains('computer science', case=False),
    Development=education['UndergradMajor'].str.contains('development', case=False),
    OtherEng=education['UndergradMajor'].str.contains('another engineering', case=False),
    NeverDeclared=education['UndergradMajor'].str.contains('never declared', case=False)
).copy()
education = education.melt(id_vars=['Year'], value_vars=['SocialScience', 'NaturalScience', 'ComputerScience',
                                                         'Development', 'OtherEng', 'NeverDeclared'],
                                                         var_name = 'EduCat', value_name = 'EduFlag')
education.dropna(how='any', inplace=True)
education = education.groupby(['Year', 'EduCat']).count().reset_index()
educationFig = sns.catplot(x='Year', y='EduFlag', col='EduCat',
                           data=education, kind='bar', height=6, aspect=1.5)

educationFig.set_xticklabels(rotation=45, ha='right')

educationFig = sns.catplot(x='Year', y='EduFlag', col='EduCat',
                           data=education, kind='bar', height=6, aspect=1.5);
educationFig.set_xticklabels(rotation=45, ha='right')

# - Data for each field
compFields = dev_data[['Year','YearsCodePro','ConvertedComp']]

D = sns.boxplot(x="Year", y="YearsCodePro",
            data=compFields)

E = sns.boxplot(x="Year", y="ConvertedComp",
            data=compFields)
imputedf = dev_data[['YearsCodePro', 'ConvertedComp']]
traind, testedf = train_test_split(imputedf, train_size=0.1)

# - Create the IterativeImputer model to predict missing values
imp = IterativeImputer(max_iter=20, random_state=0)
# - fit the model to the test database
imp.fit(imputedf)
# Transform the model on the entire dataset
compdf = pd.DataFrame(np.round(imp.transform(imputedf), 0), columns=['YearsCodePro', 'ConvertedComp'])

compdf.isnull().sum()

compPlotdf = compdf.loc[compdf['ConvertedComp'] <= 150000]
compPlotdf['CodeYearBins'] = pd.qcut(compPlotdf['YearsCodePro'], q=5)

sns.boxplot(x="CodeYearBins", y="ConvertedComp",
            data=compPlotdf)
plt.show()
