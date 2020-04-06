#!/usr/bin/env python
# coding: utf-8

# # San Francisco Crime Dataset Use Case.

#  # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

# #### @author: Martín Blázquez Moreno

# In[1]:


#Importing required packages
import re
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from IPython.core.display import display, HTML
from IPython.display import HTML
import json

import sys
sys.path.insert(0,'..')
import folium
print (folium.__file__)
print (folium.__version__)
from matplotlib.colors import Normalize, rgb2hex

import pymongo
from pymongo import MongoClient, GEO2D


# # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

# # First data exploration

# # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

# In[2]:


# Data import from csv. Name: "San Francisco crime analysis"
total_crime = pd.read_csv('Map_of_Police_Department_Incidents.csv')
print("*************")
print(total_crime.shape)# To get the number of rows and columns
print("*************")
d_crime = total_crime.head(600000) # Only the firts 600000 are saved in 
    # d_crime to reduce the computation time.
d_crime# To see the DataFrame  with 600000 rows and 12 columns


# In[3]:


#Use a 40% of the datadase and delete de total_crime file because it is so big
seed = 4
d_crime=total_crime.sample(frac=0.40, random_state=seed)
del(total_crime)
d_crime


# In[4]:


#type of each of the fields
d_crime.dtypes


# In[5]:


#To count all type of categories
d_crime['Category'].value_counts()

def plotdat(data,cat):
    l=data.groupby(cat).size()
    l.sort_values(ascending=True).sort_index()
    fig=plt.figure(figsize=(10,5))
    plt.yticks(fontsize=8)
    l.plot(kind='bar',fontsize=12,color='b')
    plt.xlabel('')
    plt.ylabel('Number of reports',fontsize=10)
    
plotdat(d_crime,'Category')


# In[6]:


# Data cleaning. Transform Data from string to date type and delta date
date=pd.to_datetime(d_crime['Date'])
print(date.min())
print(date.max())


# In[7]:


# Create a new colum "days" with timedelta format
t_delta=(date-date.min()).astype('timedelta64[D]')
d_crime['days']=t_delta
d_crime.head(1)


# ## Analyse the columns of the DataBase

# In[8]:


# Plotting bargraph. Define again.
def plotdat(data,cat):
    l=data.groupby(cat).size()
    l.sort_values(ascending=True).sort_index()
    fig=plt.figure(figsize=(10,5))
    plt.yticks(fontsize=8)
    l.plot(kind='bar',fontsize=12,color='blue')
    plt.xlabel('Type')
    plt.ylabel('Number of reports',fontsize=10)


# In[9]:


# Graph of the incident type and amount occurred
plotdat(d_crime,'Category')


# In[10]:


# Graph of the different district where the incidents occur
plotdat(d_crime,'PdDistrict')


# In[11]:


# Graph with the Day of the week when the incident occur.
plotdat(d_crime,'DayOfWeek')


# The week day is not a significant variable in terms of number of cases reported.

# In[12]:


#The greater number of cases with the same description.
l=d_crime.groupby('Descript').size()
l.sort_values()
print(l.shape)


# In[13]:


# Heatmap and hierarchical clustering
def types_districts(d_crime,per):
    
    # Group by crime type and district 
    hoods_per_type=d_crime.groupby('Descript').PdDistrict.value_counts(sort=True)
    t=hoods_per_type.unstack().fillna(0)
    
    # Sort by hood sum
    hood_sum=t.sum(axis=0)
    hood_sum.sort_values(ascending=False)
    t=t[hood_sum.index]
    
    # Filter by crime per district
    crime_sum=t.sum(axis=1)
    crime_sum.sort_values(ascending=False)
    
    # Large number, so let's slice the data.
    p=np.percentile(crime_sum,per)
    ix=crime_sum[crime_sum>p]
    t=t.loc[ix.index]
    return t


# In[14]:


t=types_districts(d_crime,98)


# In[15]:


sns.clustermap(t,cmap="mako", robust=True) 


# In[16]:


# It is required to scale dimensions
# Standardize the data within the columns (scale=1)
# Sometimes, a few values in your input have extreme values. 
# In a heatmap, this has as an effect to make every other cell the same color, what is not desired. 
# The clustermap function allows you to avoid that with the ‘robust‘ argument. Here is an example with (left) and without (right) this option.

sns.clustermap(t,standard_scale=1,cmap="mako", robust=True)


# In[17]:


# Normalize the data within the rows s_score=0
sns.clustermap(t,z_score=0,cmap="viridis", robust=True)


# # . . . . . . . . . . . . . . . . . . . . . . . . . 

# ## Let's drill down onto the Mandatory Task

# # . . . . . . . . . . . . . . . . . . . . . . . . . 

# #### Mongo Database connection and store data______________________________________________________________________________

# In[18]:


print('Mongo version', pymongo.__version__)
client = MongoClient('localhost', 27017)
db = client.test
collection = db.crimesf


# In[19]:


#Clean collection 
collection.drop()


# In[20]:


#Import data into the database. First, transform to JSON records
records = json.loads(d_crime.to_json(orient='records'))
collection.delete_many({})
collection.insert_many(records)


# In[21]:


#Check if I can access the data from the MongoDB.
cursor = collection.find().sort('Category',pymongo.ASCENDING).limit(5)
for doc in cursor:
    print(doc)


# In[22]:


#########################################################################
#T here are many categories but the analysis will be focus in VANDALISM.#
#########################################################################

#stablish a pipeline to select all rows matching attribute "Category" = "VANDALISM"
pipeline = [
        {"$match": {"Category":"VANDALISM"}},
]


# In[23]:


#We will be count the number of rows that has the category VANDALISM in our 40% data set. 
collection.count_documents({"Category":"VANDALISM"})


# In[24]:


#Query the collection with the pipeline filter. 
aggResult = collection.aggregate(pipeline)
df2 = pd.DataFrame(list(aggResult))
df2.head()


# In[25]:


# Let's have a look on Vandalism incidents' descriptions from greater amount to smaller amount
c=df2['Descript'].value_counts()
c.sort_values(ascending=False)
c.head(25)


# In[26]:


# Organize incidents' descriptions versus Districts where they were detected
def types_districts(d_crime,per):
    
    # Group by crime type and district 
    hoods_per_type=d_crime.groupby('Descript').PdDistrict.value_counts(sort=True)
    t=hoods_per_type.unstack().fillna(0)
    
    # Sort by hood sum
    hood_sum=t.sum(axis=0)
    hood_sum.sort_values(ascending=False)
    t=t[hood_sum.index]
    
    # Filter by crime per district
    crime_sum=t.sum(axis=1)
    crime_sum.sort_values(ascending=False)
    
    # Large number, so let's slice the data.
    p=np.percentile(crime_sum,per)
    ix=crime_sum[crime_sum>p]
    t=t.loc[ix.index]
    return t


# In[27]:


# Filter outliers up to 65 percentile to analyze just the vandalism type with more incidents, the rest are irrelevant
t=types_districts(df2,70)


# In[28]:


# Inspect data by means of clustermaps
sns.clustermap(t)


# In[29]:


sns.clustermap(t,standard_scale=1)


# In[30]:


sns.clustermap(t,standard_scale=0, annot=True)


# # . . . . . . . . . . . . . . . . . 

# ## Time Series Analysis

# # . . . . . . . . . . . . . . . . .

# In[31]:


# Bin crime by 30 day window. That is, obtain new colum with corresponding months 
df2['Month']=np.floor(df2['days']/30) # Approximate month (30 day window)


# In[32]:


# Default
district='All'


# In[33]:


def timeseries(dat,per):
    ''' Category grouped by month '''
    
    # Group by crime type and district 
    cat_per_time=dat.groupby('Month').Descript.value_counts(sort=True)
    t=cat_per_time.unstack().fillna(0)
  
    # Filter by crime per district
    crime_sum=t.sum(axis=0)
    crime_sum.sort_values()
    
    # Large number, so let's slice the data.
    p=np.percentile(crime_sum,per)
    ix=crime_sum[crime_sum>p]
    t=t[ix.index]
    
    return t


# In[34]:


# Filter outliers up to 10 percentile
t_all=timeseries(df2,10)


# In[35]:


#Find inciden's descriptions related to word patter "MALICIOUS"
pat = re.compile(r'MALICIOUS', re.I)


# In[36]:


pipeline = [
        {"$match": {"Category":"VANDALISM" , 'Descript': {'$regex': pat}}},
]


# In[37]:


aggResult = collection.aggregate(pipeline)
df3 = pd.DataFrame(list(aggResult))
df3.head()


# In[38]:


malicious = df3.groupby('Descript').size()
s = pd.Series(malicious)


# In[39]:


print(s)


# In[40]:


s = s[s != 1]


# In[41]:


malicious_features = list(s.index)


# In[42]:


print(malicious_features)


# In[43]:


#Let's generate a function to constructu subsets of descriptions according to patterns: 
def descriptionsAccordingToPattern(pattern):
    pat = re.compile(pattern, re.I)
   
    pipeline = [
            {"$match": {"Category":"VANDALISM" , 'Descript': {'$regex': pat}}},
    ]
    
    aggResult = collection.aggregate(pipeline)
    df3 = pd.DataFrame(list(aggResult))
    vandalism = df3.groupby('Descript').size()
    s = pd.Series(vandalism)
    s = s[s != 1] # filter those descriptions with value less equal 1
    features = list(s.index)
    
    return features


# In[44]:


# Filter by pattern 'VEHICLES'
graffiti_features = descriptionsAccordingToPattern('GRAFFITI')


# In[45]:


print(graffiti_features)


# In[46]:


suspect_features = descriptionsAccordingToPattern('ADULT SUSPECT')
break_features = descriptionsAccordingToPattern('BREAKING WINDOWS')
gun_features = descriptionsAccordingToPattern('BREAKING WINDOWS WITH BB GUN')
build_features = descriptionsAccordingToPattern('BUILDING UNDER CONSTRUCTION')
calls_features = descriptionsAccordingToPattern('FICTITIOUS PHONE CALLS')
graf_features = descriptionsAccordingToPattern('GRAFFITI')
juvenile_features = descriptionsAccordingToPattern('JUVENILE SUSPECT')
street_features = descriptionsAccordingToPattern('STREET CARS/BUSES')
tire_features = descriptionsAccordingToPattern('TIRE SLASHING')
vandalism_features = descriptionsAccordingToPattern('VANDALISM')
vehicles_features = descriptionsAccordingToPattern('VANDALISM OF VEHICLES')


# In[47]:


# Lets use real dates for plotting
days_from_start=pd.Series(t_all.index*30).astype('timedelta64[D]')
dates_for_plot=date.min()+days_from_start
time_labels=dates_for_plot.map(lambda x: str(x.year)+'-'+str(x.month))


# In[48]:


# Analytics per vandalism tipology according to descriptions
def vandalism_analysis(t,district,plot):
    t['ADULT SUSPECT']=t[suspect_features].sum(axis=1)
    t['BREAKING WINDOWS']=t[break_features].sum(axis=1)
    t['BREAKING WINDOWS WITH BB GUN']=t[gun_features].sum(axis=1)
    t['BUILDING UNDER CONSTRUCTION']=t[build_features].sum(axis=1)
    t['FICTITIOUS PHONE CALLS']=t[calls_features].sum(axis=1)
    t['GRAFFITI']=t[graf_features].sum(axis=1)
    t['JUVENILE SUSPECT']=t[juvenile_features].sum(axis=1)
    t['STREET CARS/BUSES']=t[street_features].sum(axis=1)
    t['TIRE SLASHING']=t[tire_features].sum(axis=1)
    t['VANDALISM']=t[vandalism_features].sum(axis=1)
    t['VANDALISM OF VEHICLES']=t[vehicles_features].sum(axis=1)
    
    vandalism=t[['ADULT SUSPECT','BREAKING WINDOWS','BREAKING WINDOWS WITH BB GUN','BUILDING UNDER CONSTRUCTION','FICTITIOUS PHONE CALLS','GRAFFITI','JUVENILE SUSPECT','STREET CARS/BUSES','TIRE SLASHING','VANDALISM','VANDALISM OF VEHICLES']]
    if plot:
        vandalism.index=[int(i) for i in vandalism.index]
        colors = plt.cm.jet(np.linspace(0, 1, vandalism.shape[1]))
        vandalism.plot(kind='bar', stacked=True, figsize=(20,10), color=colors, width=1, title=district,fontsize=6)
    return vandalism


# In[49]:


vanda_df_all=vandalism_analysis(t_all,district,True)


# In[50]:


def vandalism_analysis_rescale(t,district,plot):
    t['ADULT SUSPECT']=t[suspect_features].sum(axis=1)
    t['BREAKING WINDOWS']=t[break_features].sum(axis=1)
    t['BREAKING WINDOWS WITH BB GUN']=t[gun_features].sum(axis=1)
    t['BUILDING UNDER CONSTRUCTION']=t[build_features].sum(axis=1)
    t['FICTITIOUS PHONE CALLS']=t[calls_features].sum(axis=1)
    t['GRAFFITI']=t[graf_features].sum(axis=1)
    t['JUVENILE SUSPECT']=t[juvenile_features].sum(axis=1)
    t['STREET CARS/BUSES']=t[street_features].sum(axis=1)
    t['TIRE SLASHING']=t[tire_features].sum(axis=1)
    t['VANDALISM']=t[vandalism_features].sum(axis=1)
    t['VANDALISM OF VEHICLES']=t[vehicles_features].sum(axis=1)

    vandalism=t[['ADULT SUSPECT','BREAKING WINDOWS','BREAKING WINDOWS WITH BB GUN','BUILDING UNDER CONSTRUCTION','FICTITIOUS PHONE CALLS','GRAFFITI','JUVENILE SUSPECT','STREET CARS/BUSES','TIRE SLASHING','VANDALISM','VANDALISM OF VEHICLES']]
    if plot:
        vandalism=vandalism.div(vandalism.sum(axis=1),axis=0)
        vandalism.index=[int(i) for i in vandalism.index]
        colors = plt.cm.GnBu(np.linspace(0, 1, vandalism.shape[1]))
        colors = plt.cm.jet(np.linspace(0, 1, vandalism.shape[1]))
        vandalism.plot(kind='bar', stacked=True, figsize=(20,10), color=colors, width=1, title=district, legend=True)
        plt.ylim([0,1])
    return vandalism


# In[51]:


vanda_df_all=vandalism_analysis_rescale(t_all,district,True)


# # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

# # Focussing on real dates and Districts

# # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

# #### Let's add the real dates.
# #### And focusing on several types of vandalism

# In[52]:


dates_for_plot.index=dates_for_plot
sns.set_context(rc={"figure.figsize": (25.5,5.5)})
for d,c in zip(['GRAFFITI','BREAKING WINDOWS','STREET CARS/BUSES','TIRE SLASHING'],['b','r','c','g']):
    plt.plot(dates_for_plot.index,vanda_df_all[d],'o-',color=c,ms=6,mew=1.5,mec='white',linewidth=0.5,label=d,alpha=0.75)
plt.legend(loc='upper left',scatterpoints=1,prop={'size':8})


# In[53]:


# filter in 2006 because there are outlier between 2005 and 2006
dates_for_plot.index=dates_for_plot
sns.set_context(rc={"figure.figsize": (10,1)})
for d,c in zip(['GRAFFITI','BREAKING WINDOWS','STREET CARS/BUSES','TIRE SLASHING'],['b','r','c','g']):
    plt.plot(dates_for_plot.tail(140).index,vanda_df_all[d].tail(140),'o-',color=c,ms=6,mew=1.5,mec='white',linewidth=0.5,label=d,alpha=0.75)
plt.legend(loc='upper left',scatterpoints=1,prop={'size':8})


# In[54]:


dates_for_plot.index=dates_for_plot
sns.set_context(rc={"figure.figsize": (5,5)})
for d,c in zip(['GRAFFITI','BREAKING WINDOWS','STREET CARS/BUSES','TIRE SLASHING'],['b','r','c','g']):
    plt.plot(dates_for_plot.tail(140).index,vanda_df_all[d].tail(140),'o-',color=c,ms=6,mew=1.5,mec='white',linewidth=0.5,label=d,alpha=0.75)
plt.legend(loc='upper left',scatterpoints=1,prop={'size':8})


# In[55]:


# Remove 'TIRE SLASHING' and  'STREET CARS/BUSES' as it has different range
dates_for_plot.index=dates_for_plot
sns.set_context(rc={"figure.figsize": (25.5,5.5)})
for d,c in zip(['GRAFFITI','BREAKING WINDOWS'],['b','r']):
    plt.plot(dates_for_plot.tail(140).index,vanda_df_all[d].tail(140),'o-',color=c,ms=6,mew=1.5,mec='white',linewidth=0.5,label=d,alpha=0.75)
plt.legend(loc='upper left',scatterpoints=1,prop={'size':8})


# ### To see more in depth, iterate through each district.

# In[56]:


#Group by the vandalism by districts in the diferents graph
stor=[]
stor_time=[]

for d in d_crime['PdDistrict'].value_counts().index:
    # Specify district and group by time
    dist_dat=df2[df2['PdDistrict']==d]
    t=timeseries(dist_dat,11)
    # Merge to ensure all categories are preserved!
    t_merge=pd.DataFrame(columns=t_all.columns)
    m=pd.concat([t_merge,t],axis=0).fillna(0)
    m.reset_index(inplace=True)
    # Plot
    vanda_df=vandalism_analysis(m,d,True)
    plt.show()
    s=vanda_df.sum(axis=0)
    stor=stor+[s]
    vanda_df.columns=cols=[c+"_%s"%d for c in vanda_df.columns]
    stor_time=stor_time+[vanda_df]
    
vanda_dat_time=pd.concat(stor_time,axis=1)
vanda_dat=pd.concat(stor,axis=1)
vanda_dat.columns=[d_crime['PdDistrict'].value_counts().index]


# ## Let's perform Correlation Analysis

# In[57]:


##We can also look at correlations between areas for different types vandalism.

sns.set_context(rc={"figure.figsize": (20,20)})
corr = vanda_dat_time.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(19, 19))

# Generate a custom diverging colormap
sns.set_context(rc={"figure.figsize": (20,20)})
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Plot the correlation heatmap
sns.heatmap(corr,  mask=mask, cmap=cmap, vmax=0.5, center=0, square=True, linewidths=.5, cbar_kws={"shrink": 0.5})


# ### Correlation Analysis

# In[58]:


#With this in mind, we can examine select timeseries data.
vanda_dat_time.index=dates_for_plot.head(181)
sns.set_context(rc={"figure.figsize": (7.5,5)})
for d,c in zip(['BREAKING WINDOWS_MISSION','GRAFFITI_MISSION'],['b','r']):
    plt.plot(vanda_dat_time.index,vanda_dat_time[d],'o-',color=c,ms=6,mew=1,mec='white',linewidth=0.5,label=d,alpha=0.75)
plt.legend(loc='upper left',scatterpoints=1,prop={'size':8})


# In[59]:


vanda_dat_time.index=dates_for_plot.head(181)
sns.set_context(rc={"figure.figsize": (30,15)})
for d,c in zip(['BREAKING WINDOWS_MISSION','GRAFFITI_MISSION','STREET CARS/BUSES_MISSION','TIRE SLASHING_CENTRAL'],['b','r','g','c']):
    plt.plot(vanda_dat_time.index,vanda_dat_time[d],'o-',color=c,ms=5,mew=1,mec='white',linewidth=0.5,label=d,alpha=0.75)
plt.legend(loc='upper left',scatterpoints=1,prop={'size':10})


# # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

# ## San Francisco Incidents: Analysis by neighborhood

# ### Now we examine SPATIAL Relationships (Advance)

# # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

# In[60]:


# We will analyze more in deep. Splitting data by district

stor=[]
stor_time=[]


for d in d_crime['PdDistrict'].value_counts().index:

    # Specify district and group by time
        dist_dat=df2[df2['PdDistrict']==d]
        t=timeseries(dist_dat,10)
        
        # Merge to ensure all categories are preserved!
        t_merge=pd.DataFrame(columns=t_all.columns)
        m=pd.concat([t_merge,t],axis=0).fillna(0)
        m.reset_index(inplace=True)

        # Plot
        vanda_df=vandalism_analysis(m,d,True)
        plt.show()
        s=vanda_df.sum(axis=0)
        stor=stor+[s]
        vanda_df.columns=cols=[c+"_%s"%d for c in vanda_df.columns]
        stor_time=stor_time+[vanda_df]

    

vanda_dat_time=pd.concat(stor_time,axis=1)
vanda_dat=pd.concat(stor,axis=1)
vanda_dat.columns=[d_crime['PdDistrict'].value_counts().index]


# In[61]:


#We can now summarize this data using clustered heatmaps.
sns.clustermap(vanda_dat,standard_scale=1,cmap="viridis",robust=True,annot=True)


# ### Mapping relationships

# In[62]:


#Let's isolate all vandalism-related records.

tmp=df2.copy()
tmp.set_index('Descript',inplace=True)

vandalism_dat=tmp.loc[vandalism_features]
vandalism_pts=vandalism_dat[['X','Y','Month']]


# In[63]:


print(vandalism_pts)


# In[64]:


#Plot the suspect regimes.

d=pd.DataFrame(vandalism_pts.groupby('Month').size())
d.index=dates_for_plot.head(181)
d.columns=['Count']

diff=len(d.index)-120


# In[65]:


print(diff)


# In[66]:


plt.plot(d.index,d['Count'],'o-',color='k',ms=6,mew=1,mec='white',linewidth=0.5,label=d,alpha=0.75)
plt.axvspan(d.index[40-diff],d.index[40],color='cyan',alpha=0.5)
plt.axvspan(d.index[80-diff],d.index[80],color='red',alpha=0.5)
plt.axvspan(d.index[120],d.index[-1],color='green',alpha=0.5)


# In[67]:


oldest_vandalism_sums=d.loc[(d.index>d.index[40-diff]) & (d.index<d.index[40])]
old_vandalism_sums=d.loc[(d.index>d.index[80-diff]) & (d.index<d.index[80])]
new_vandalism_sums=d.loc[d.index>d.index[120]]


# In[68]:


#Fold-difference in mean between the two regimes.
old_vandalism_sums['Count'].mean()/float(new_vandalism_sums['Count'].mean())


# In[69]:


#Two regimes.

oldest_vandalism=vandalism_pts[(vandalism_pts['Month']>(40-diff)) & (vandalism_pts['Month']<40)]
oldest_vandalism.columns=['longitude','latitude','time']
old_vandalism=vandalism_pts[(vandalism_pts['Month']>(80-diff)) & (vandalism_pts['Month']<80)]
old_vandalism.columns=['longitude','latitude','time']
new_vandalism=vandalism_pts[vandalism_pts['Month']>120]
new_vandalism.columns=['longitude','latitude','time']


# ## We can look at this spatially.
# ##### Use a shapefile for Neighborhoods in SF to overlay the data onto a map.
# ##### https://data.sfgov.org/Geographic-Locations-and-Boundaries/Neighborhoods/ejmn-jyk6
# ##### Basemap can be used to view this. Some nice work at this link that I drew from:
# ##### http://sensitivecities.com/so-youd-like-to-make-a-map-using-python-EN.html

# In[70]:


# Let's create MongoBD collections to manage and query data
col1 = db.vandalismn
col2 = db.vandalismo
#Import data into the database
col1.drop()
col2.drop()


# In[71]:


# Collection for new suspect
data_json1 = json.loads(new_vandalism.to_json(orient='records'))
col1.delete_many({})
col1.insert_many(data_json1)


# In[72]:


# Collection for old suspect
data_json2 = json.loads(old_vandalism.to_json(orient='records'))
col2.delete_many({})
col2.insert_many(data_json2)


# In[73]:


#Check if you can access the data from the MongoDB.
cursor = col1.find().sort('time',pymongo.ASCENDING).limit(3)
for doc in cursor:
    print(doc)


# In[74]:


cursor2 = col2.find().sort('time',pymongo.ASCENDING).limit(10)
for doc in cursor2:
    print(doc)


# In[75]:


# Create a new collection to store districts geo points
col3 = db.districts
col3.drop()


# In[76]:


# mongoBD import from geojson file containing geopoints in form of multipopygons
os.system('"mongoimport" -d test -c districts --file \\Analysis_Neighborhoods.geojson')


# In[77]:


cursor3 = col3.find().limit(10)
#for doc in cursor3:
    #print(doc)
    
# I commented the loop because Github show every row (there are so much rows :D )


# In[78]:


# Get information about the index
col2.index_information()


# ### Prepare data to perform spatial queries by Mongo

# In[79]:


col_temp = db.vandalismn2
cursor = col2.find()
for doc in cursor:
    col_temp.insert_one({
        "loc": {
            "type": "Point",
            "coordinates": [doc["longitude"], doc["latitude"]]
        }
    });


# In[80]:


cursor = col_temp.find().limit(10)
for doc in cursor:
    print(doc)


# In[81]:


cursor2 = col2.find_one()   


# In[82]:


query = {"features.geometry": 
    { "$geoIntersects": 
                { "$geometry": 
                    {"type": "Point", 
                     "coordinates": [cursor2["longitude"],cursor2["latitude"]] 
                    }
                } 
            } 
        }


# In[83]:


cursor3 = col3.find_one(query)


# In[84]:


# Generate new collection with features filteres
collection_features = db.feat
collection_features.delete_many({})
collection_features.insert_many(cursor3["features"])


# In[85]:


# Obtain cursor
cursor4 = collection_features.find_one()


# In[86]:


# Spatial query implementing getoIntersects operation
query_feat = {"geometry": 
                { "$geoIntersects": 
                { "$geometry": 
                    {"type": "Point", 
                     "coordinates": [cursor2["longitude"],cursor2["latitude"]] 
                    }
                } 
            } 
        }


# In[87]:


# Have a look if every thing is OK

#for doc in collection_features.find(query_feat):
#    print(doc)

# I commented this as same before.(github show everything)


# In[88]:


# Obtain the selected neighborhood
cursor_feat = collection_features.find_one(query_feat)
print(cursor_feat["properties"])


# ### Printing data in Maps

# In[89]:


# Set general coordinates of San Francisco city
SF_COORDINATES = (37.76, -122.45) ## San Francisco Coordinates


# In[90]:


MAX_RECORDS = 100
m = folium.Map(location=SF_COORDINATES, zoom_start=12)


# In[91]:


#Display neighborhoods by polygons
geo_json_data = json.load(open('Analysis_Neighborhoods.geojson'))
folium.GeoJson(geo_json_data).add_to(m)


# In[92]:


# Display Old VANDALISM points (Red)
cursor = col1.find().limit(MAX_RECORDS)
for doc in cursor:
    folium.Marker(location = [doc["latitude"],doc["longitude"]],
                  popup='Old Vandalism',
                  icon=folium.Icon(color='red')).add_to(m)


# In[93]:


# Display New VANDALISM Points (Green)
cursor = col2.find().limit(MAX_RECORDS)
for doc in cursor:
    folium.Marker(location = [doc["latitude"],doc["longitude"]],
                  popup='New Vandalism',
                  icon=folium.Icon(color='green')).add_to(m)


# In[94]:


# Display queried point in spatial query (Blue)
folium.Marker(location = [cursor2["latitude"],cursor2["longitude"]],
                  popup='Selected Point',
                  icon=folium.Icon(color='blue')).add_to(m)


# In[95]:


folium.LayerControl().add_to(m)


# In[96]:


m.save(outfile='map1.html')


# In[97]:


#Map with the location of the vandalism
m


# In[98]:


# Obtaining the geometry of an Incident


# In[99]:


map2 = folium.Map(location=SF_COORDINATES, zoom_start=12)


# In[100]:


folium.GeoJson(
    cursor_feat["geometry"],
    name='Selected Neighborhood'
).add_to(map2)


# In[101]:


folium.Marker(location = [cursor2["latitude"],cursor2["longitude"]],
                  popup='Selected Point',
                  icon=folium.Icon(color='blue')).add_to(map2)


# In[102]:


folium.LayerControl().add_to(map2)


# In[103]:


map2.save(outfile='map2.html')


# In[104]:


map2


# ### To detect every vandalism (new and old) in a especific district:

# In[105]:


query_hood = {"loc": 
                { "$geoIntersects": 
                { "$geometry": 
                    {"type": "MultiPolygon", 
                     "coordinates": cursor_feat["geometry"]["coordinates"]
                    }
                }
            }
        }


# In[106]:


col_temp = db.vandalismn2
cursor = col2.find()
for doc in cursor:
    col_temp.insert_one({
        "loc": {
            "type": "Point",
            "coordinates": [doc["longitude"], doc["latitude"]]
        }
    });


# In[107]:


cursor = col_temp.find()
col_temp2 = db.vandalismo2
cursor = col1.find()
for doc in cursor:
    col_temp2.insert_one({
        "loc": {
            "type": "Point",
            "coordinates": [doc["longitude"], doc["latitude"]]
        }
    });


# In[108]:


map3 = folium.Map(location=SF_COORDINATES, zoom_start=12)

folium.GeoJson(
    cursor_feat["geometry"],
    name='Selected Neighborhood'
).add_to(map3)


# In[109]:


cursor_p = col_temp.find(query_hood).limit(70)

for doc in cursor_p:
    folium.Marker(location = [doc["loc"]["coordinates"][1],doc["loc"]["coordinates"][0]],
                  popup='New Vandalism',
                  icon=folium.Icon(color='green')).add_to(map3)


# In[110]:


cursor_p2 = col_temp2.find(query_hood).limit(70)

for doc in cursor_p2:
    folium.Marker(location = [doc["loc"]["coordinates"][1],doc["loc"]["coordinates"][0]],
                  popup='Old Vandalism',
                  icon=folium.Icon(color='red')).add_to(map3)


# In[111]:


folium.Marker(location = [cursor2["latitude"],cursor2["longitude"]],

                  popup='Selected Point',
                  icon=folium.Icon(color='blue')).add_to(map3)

folium.LayerControl().add_to(map3)
map3.save(outfile='map3.html')


# In[112]:


map3

