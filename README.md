# Data Science: Spanish train analysis overview

- Create virtualization to have a better understanding of the data in Spanish train.
- Feature engineering form the original to create much better virtualization.
- Building interactive plot by using plotly which might take more resources but have better interactive to use.

## Code and Resources Used
**Language:** Python for virtualization, and R for machine learning.
**Packages:** numpy,pandas,seaborn,matplotlib,plotly,sklearn.

## Data cleaning
After investing time in analysing what we could do with the data, I've so many ideas that could do much more, for example.
- Changing some variables data type.
- Remove outliers that have negative effect on virtualization
- Column the time to calculate the price and others more efficiency
- Remove Null and fill the most suitable value.

**Example:** insert_date, start_data, end_date into the separate year, month and date type for better understanding.
```Python
for col in ['insert_date', 'start_date', 'end_date']:
    date_col = pd.to_datetime(df[col])
    df[col + '_hour'] = date_col.dt.hour
    df[col + '_minute'] = date_col.dt.minute
    df[col + '_second'] = date_col.dt.second
    df[col + '_day'] = date_col.dt.day
    df[col + '_weekday'] = date_col.dt.day_name()
    df[col + '_month'] = date_col.dt.month
    df[col + '_year'] = date_col.dt.year

for i in ['insert_date','start_date','end_date']:
    df[i] = pd.to_datetime(df[i])
```

## Feature Engineering
I've created new columns based on past data such as is_journey_end_on_sameday, travel_time_in_mins and more for further analysis.
```Python
df['is_journey_end_on_sameday'] = np.where(df['start_date'].dt.date==df['end_date'].dt.date, 1, 0)
df['travel_time_in_mins'] = df['end_date'] - df['start_date']
df['travel_time_in_mins']=df['travel_time_in_mins']/np.timedelta64(1,'m')
```

## Data Exploration and Virtualization
To understand the model that we want to predict better, virtualization is a must! So I've done fundamental analysis to understand it better after cleaning and manipulating. 

The better thing in this notebook I've started to use Plotly for better virtualization but take more resources than Matplotlib. Let's check some examples from Plotly and Matplotlib.

**Example of Matplotlib**
Checking the correlation between variable.
![Correlation](https://github.com/northpr/SpanishTrain/blob/main/images/Screen%20Shot%202564-10-29%20at%2015.26.24.png)
```Python
plt.subplots(figsize=(10,10))
sns.set(font_scale=1)
hm = sns.heatmap(cm,annot=True,yticklabels=top_corr.values, xticklabels=top_corr.values)
plt.show()
```

We could know almost everyone people use AVE train type more than any other types.
![Distribution](https://github.com/northpr/SpanishTrain/blob/main/images/Screen%20Shot%202564-10-29%20at%2015.29.44.png)
```Python
f,ax = plt.subplots(figsize=(15,6))
ax = sns.distplot(df['price'],rug=True,)
plt.show()
```

"Turista con enlance" has the lowest ticket price 
but "Cama G Clase" has the highest ticket price.
![BoxPlot](https://github.com/northpr/SpanishTrain/blob/main/images/Screen%20Shot%202564-10-29%20at%2015.30.09.png)
```Python
f,ax = plt.subplots(figsize=(15,6))
ax = sns.boxplot(x='train_type',y='price',data=df)
plt.show()
```

**Example of Plotly** For better interaction, I just want to try it because it looks fun!
Pie chart to Train types
![Distribution](https://github.com/northpr/SpanishTrain/blob/main/images/newplot.png?raw=true)
```Python
countpie = df['train_type'].value_counts()

fig = {
  "data": [
    {
      "values": countpie.values,
      "labels": countpie.index,
      "domain": {"x": [0, .5]},
      "name": "Train types",
      "hoverinfo":"label+percent+name",
      "hole": .7,
      "type": "pie"
    },],
  "layout": {
        "title":"Pie chart Train types",
    }
}
iplot(fig)
```

![Histogram](https://github.com/northpr/SpanishTrain/blob/main/images/newplot-2.png?raw=true)
```Python
px.histogram(df, x="train_type",y="travel_time_in_mins" ,color="train_type")
```

Average travel time in minutes by Train type
![Scatter](https://github.com/northpr/SpanishTrain/blob/main/images/newplot-3.png?raw=true)
```Python
plotter = df.groupby('train_type')['travel_time_in_mins'].agg(['mean'])
plotter.columns = ["mean"]
plotter['train_type'] = plotter.index

data = [
    {
        'x': plotter['train_type'],
        'y': plotter['mean'],
        'mode': 'markers+text',
        'text' : plotter['train_type'],
        'textposition' : 'bottom center',
        'marker': {  
            'size': 20,
        }
    }
]

layout = go.Layout(title="Average travel time in minutes by Train type", 
                   xaxis=dict(title='Train type'),
                   yaxis=dict(title='Travel time in minutes')
                  )
fig = go.Figure(data = data, layout = layout)
iplot(fig, filename='scatter0')
```

## Model Building
I've not done much for this part, just making basic predictions such as changing categorical variables into dummy variables. Also, split the data into train and test.

The model is not good enough to use further because I've not tuned and done more, but you take a quick look at it.

My R-squared is at 0.627.

## Final
This notebook is the best to take a look in data analyze and basic data manipulation. You can check more by downloading my notebook or html file above.

Thanks for viewing please leave a star for me! :)
