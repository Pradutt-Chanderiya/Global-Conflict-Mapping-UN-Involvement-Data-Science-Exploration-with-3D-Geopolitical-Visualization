#!/usr/bin/env python
# coding: utf-8

# # Install Necessary Libraries

# In[1]:


# Install required libraries
get_ipython().system('pip install pandas numpy matplotlib seaborn plotly scikit-learn geopandas folium dash pycaret tensorflow keras xgboost pyspark lightgbm apache-spark apache-airflow apache-kafka nltk gensim shapely statsmodels mapbox scipy dask')


# #  Import Libraries

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import geopandas as gpd
import folium
import shapely
import nltk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras
import xgboost as xgb
import lightgbm as lgb


# # Generate and Load a Custom Dataset

# In[43]:


# Create mock dataset with equal-length lists (11 entries each)
data = {
    'Country': ['USA', 'Russia', 'India', 'China', 'Ukraine', 'Israel', 'Palestine', 'France', 'Germany', 'Pakistan', 'Taiwan'],
    'Conflict_Region': ['Mid North America', 'Eastern Europe', 'South Asia', 'East Asia', 'Eastern Europe', 'Middle East', 'Middle East', 'Western Europe', 'Western Europe', 'South West Asia', 'East Asia'],
    'Conflict_Type': ['Tension', 'War', 'Tension', 'Potential Conflict', 'War', 'War', 'War', 'Potential Conflict', 'Tension', 'Extreme Tension', 'Potential Conflict'],
    'Latitude': [37.0902, 61.5240, 20.5937, 35.8617, 48.3794, 31.0461, 31.9522, 46.6034, 51.1657, 30.3753, 23.6978],
    'Longitude': [-95.7129, 105.3188, 78.9629, 104.1954, 31.1656, 34.8516, 35.2332, 1.8883, 10.4515, 69.3451, 121.0200],
    'Altitude': [760, 600, 160, 1840, 175, 508, 795, 375, 263, 900, 1150],
    'Conflict_Intensity': [10, 20, 30, 25, 15, 18, 5, 12, 22, 16, 14],
    'Deaths': np.random.randint(1000, 50000, size=11),  # Adjusted to 11 entries
    'Economic_Impact_Billion': np.random.uniform(1.5, 100, size=11),  # Adjusted to 11 entries
    'Environmental_Damage_Index': np.random.uniform(1, 10, size=11),  # Adjusted to 11 entries
    'UN_Interventions': np.random.choice([1, 2, 3, 4], size=11),  # Adjusted to 11 entries
    'Total_Population': [331002651, 145912025, 1380004385, 1439323776, 43733762, 8655535, 5000000, 65273511, 83783942, 225199937, 23816775],
    'Male_Population': [162000000, 67000000, 705000000, 724000000, 22000000, 4300000, 2500000, 32000000, 41000000, 113000000, 12000000],
    'Female_Population': [169000000, 78900000, 675000000, 715000000, 21700000, 4350000, 2500000, 33200000, 42700000, 112000000, 11800000]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv(r'E:\conflict_data.csv', index=False)

# Set display options to show all rows and columns
pd.set_option('display.max_rows', None)  # Display all rows
pd.set_option('display.max_columns', None)  # Display all columns

# Display the full DataFrame
print(df)

# Display first few rows
df.head()


# # Basic Data Exploration

# In[44]:


df.info()
df.describe()


# # Data Cleaning

# In[45]:


# Checking for missing values
print(df.isnull().sum())

# Handling missing values (if any)
df.fillna(0, inplace=True)


# # Data Wrangling

# In[36]:


# Convert columns to appropriate types if necessary
df['Deaths'] = df['Deaths'].astype(int)
df['Economic_Impact_Billion'] = df['Economic_Impact_Billion'].astype(float)


# # Bar Chart: Conflict Types by Country
# 

# In[14]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Country', hue='Conflict_Type')
plt.title('Conflict Types by Country')
plt.xticks(rotation=45)
plt.show()


# # Heatmap: Correlation of Economic and Environmental Impact
# 

# In[15]:


corr = df[['Deaths', 'Economic_Impact_Billion', 'Environmental_Damage_Index']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Deaths, Economic Impact, and Environmental Damage')
plt.show()


# # Pie Chart: Distribution of UN Interventions
# 

# In[16]:


df['UN_Interventions'].value_counts().plot.pie(autopct='%1.1f%%', figsize=(8, 8))
plt.title('UN Interventions Distribution')
plt.ylabel('')
plt.show()


# # 3D Geopolitical Map with Plotly and Folium
# 

# In[1]:


import pandas as pd
import plotly.graph_objects as go

# Sample data
data = {
    'Country': ['USA', 'Russia', 'India', 'China', 'Ukraine', 'Israel', 'Palestine', 'France', 'Germany', 'Pakistan', 'Taiwan'],
    'Conflict_Type': ['Tension', 'War', 'Tension', 'Potential Conflict', 'War', 'War', 'War', 'Potential Conflict', 'Tension', 'Extreme Tension', 'Potential Conflict']
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Coordinates for the countries (latitude and longitude)
coordinates = {
    'USA': (37.0902, -95.7129),
    'Russia': (61.5240, 105.3180),
    'India': (20.5937, 78.9629),
    'China': (35.8617, 104.1954),
    'Ukraine': (48.3794, 31.1656),
    'Israel': (31.0461, 34.8516),
    'Palestine': (31.9522, 35.2332),
    'France': (46.6034, 1.8883),
    'Germany': (51.1657, 10.4515),
    'Pakistan': (30.3753, 69.3451),
    'Taiwan': (23.6978, 120.9605)
}

# Adding coordinates to the DataFrame
df['Latitude'] = df['Country'].map(lambda country: coordinates[country][0])
df['Longitude'] = df['Country'].map(lambda country: coordinates[country][1])

# Color mapping for Conflict_Type
color_map = {
    'Tension': 'yellow',
    'War': 'red',
    'Potential Conflict': 'orange',
    'Extreme Tension': 'purple'
}

# Create a scatter plot for the circles
fig = go.Figure()

for index, row in df.iterrows():
    fig.add_trace(go.Scattergeo(
        lon=[row['Longitude']],
        lat=[row['Latitude']],
        text=row['Country'],
        mode='markers+text',
        marker=dict(
            size=12,
            color=color_map[row['Conflict_Type']],
            line=dict(width=2),  # Thicker outline for the markers
            opacity=0.7
        ),
        textposition="top center",
        textfont=dict(
            family="Arial",
            size=12,
            color="black"  # Set country names to extreme black
        )
    ))

# Add custom legend with small colored boxes
annotations = [
    dict(
        x=0.95, y=1,  # Coordinates of the legend box in the top-right corner
        xref='paper', yref='paper',
        showarrow=False,
        text='<b>Legend</b>',
        font=dict(size=14, color='black'),
        align='left'
    ),
    dict(
        x=0.95, y=0.95,  # Position for 'Tension'
        xref='paper', yref='paper',
        showarrow=False,
        text='Tension',
        font=dict(size=12, color='black'),
        bgcolor='yellow',  # Box color for 'Tension'
        bordercolor='black',
        align='left'
    ),
    dict(
        x=0.95, y=0.90,  # Position for 'War'
        xref='paper', yref='paper',
        showarrow=False,
        text='War',
        font=dict(size=12, color='black'),
        bgcolor='red',  # Box color for 'War'
        bordercolor='black',
        align='left'
    ),
    dict(
        x=0.95, y=0.85,  # Position for 'Potential Conflict'
        xref='paper', yref='paper',
        showarrow=False,
        text='Potential Conflict',
        font=dict(size=12, color='black'),
        bgcolor='orange',  # Box color for 'Potential Conflict'
        bordercolor='black',
        align='left'
    ),
    dict(
        x=0.95, y=0.80,  # Position for 'Extreme Tension'
        xref='paper', yref='paper',
        showarrow=False,
        text='Extreme Tension',
        font=dict(size=12, color='black'),
        bgcolor='purple',  # Box color for 'Extreme Tension'
        bordercolor='black',
        align='left'
    )
]

# Update layout for the map
fig.update_layout(
    title='Global Conflict Mapping',
    geo=dict(
        scope='world',
        showland=True,
        landcolor='grey',  # Color of the land
        oceancolor='blue',  # Color of the oceans
        countrycolor='black',  # Color of the country boundaries
        coastlinecolor='black',
        projection_type='natural earth',
        center={"lat": 20, "lon": 0},  # Center of the map
        projection_scale=3.5,  # Increase/decrease size of the map
    ),
    height=700,
    width=1200,
    annotations=annotations  # Add the annotations for the legend
)

# Show the figure
fig.show()


# In[19]:


import pandas as pd
import folium
from folium.plugins import MarkerCluster
import plotly.express as px
import numpy as np
from branca.element import Template, MacroElement

# Sample data
data = {
    'Country': ['USA', 'Russia', 'India', 'China', 'Ukraine', 'Israel', 'Palestine', 'France', 'Germany', 'Pakistan', 'Taiwan'],
    'Conflict_Type': ['Tension', 'War', 'Tension', 'Potential Conflict', 'War', 'War', 'War', 'Potential Conflict', 'Tension', 'Extreme Tension', 'Potential Conflict'],
}

# Create a DataFrame
df = pd.DataFrame(data)

# Define geographical coordinates for countries
geo_coords = {
    'USA': [37.0902, -95.7129],
    'Russia': [61.5240, 105.3188],
    'India': [20.5937, 78.9629],
    'China': [35.8617, 104.1954],
    'Ukraine': [48.3794, 31.1656],
    'Israel': [31.0461, 34.8516],
    'Palestine': [31.9522, 35.2332],
    'France': [46.6034, 1.8883],
    'Germany': [51.1657, 10.4515],
    'Pakistan': [30.3753, 69.3451],
    'Taiwan': [23.6978, 120.9605]
}

# Define colors for each country based on conflict type
conflict_colors = {
    'Tension': 'yellow',
    'War': 'red',
    'Potential Conflict': 'orange',
    'Extreme Tension': 'purple',
}

# Initialize Folium map centered at a global perspective
m = folium.Map(location=[20, 0], zoom_start=2, control_scale=True)

# Define ocean and land color using Tile Layer
folium.TileLayer('CartoDB positron').add_to(m)

# Add circles, labels and tooltips with country names in extreme black color
for index, row in df.iterrows():
    country = row['Country']
    conflict_type = row['Conflict_Type']
    
    # Get coordinates for the country
    coords = geo_coords[country]
    
    # Create a circle marker for each country
    folium.CircleMarker(
        location=coords,
        radius=10,  # You can adjust the radius to increase/decrease size
        color=conflict_colors[conflict_type],
        fill=True,
        fill_opacity=0.6,
        popup=f"<b>{country}</b>: {conflict_type}",
        tooltip=folium.Tooltip(f"<span style='color:black;'><b>{country}</b></span>")  # Tooltip with country name in black color
    ).add_to(m)

# Draw country boundaries with thicker outline
folium.GeoJson(
    'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/world-countries.json',
    style_function=lambda x: {
        'fillColor': 'grey', 
        'color': 'black', 
        'weight': 2,  # Thicker outline for country borders
        'fillOpacity': 0.6,
    }
).add_to(m)

# Add a color legend to the top right corner of the map
legend_html = '''
     <div style="
     position: fixed; 
     bottom: 50px; left: 50px; width: 150px; height: 120px; 
     background-color: white; z-index:9999; font-size:14px;
     border:2px solid grey;
     ">
     &nbsp; <b>Conflict Legend</b> <br>
     &nbsp; <i class="fa fa-circle" style="color:yellow"></i> Tension<br>
     &nbsp; <i class="fa fa-circle" style="color:red"></i> War<br>
     &nbsp; <i class="fa fa-circle" style="color:orange"></i> Potential Conflict<br>
     &nbsp; <i class="fa fa-circle" style="color:purple"></i> Extreme Tension<br>
     </div>
     '''
m.get_root().html.add_child(folium.Element(legend_html))

# Display the map
m.save('3D_Geopolitical_Map_with_Legend.html')
m


# In[ ]:





# # Choropleth Map: UN Interventions by Country
# 

# In[15]:


import pandas as pd
import folium
from folium.features import GeoJsonTooltip

# Sample data
data = {
    'Country': ['USA', 'Russia', 'India', 'China', 'Ukraine', 'Israel', 'Palestine', 'France', 'Germany', 'Pakistan', 'Taiwan'],
    'Conflict_Type': ['Tension', 'War', 'Tension', 'Potential Conflict', 'War', 'War', 'War', 'Potential Conflict', 'Tension', 'Extreme Tension', 'Potential Conflict']
}

# Create a DataFrame
df = pd.DataFrame(data)

# Define geographical coordinates for the countries
geo_coords = {
    'USA': [37.0902, -95.7129],
    'Russia': [61.5240, 105.3188],
    'India': [20.5937, 78.9629],
    'China': [35.8617, 104.1954],
    'Ukraine': [48.3794, 31.1656],
    'Israel': [31.0461, 34.8516],
    'Palestine': [31.9522, 35.2332],
    'France': [46.6034, 1.8883],
    'Germany': [51.1657, 10.4515],
    'Pakistan': [30.3753, 69.3451],
    'Taiwan': [23.6978, 120.9605]
}

# Define random colors for each country for the circle markers
country_colors = {
    'USA': 'green',
    'Russia': 'blue',
    'India': 'orange',
    'China': 'purple',
    'Ukraine': 'red',
    'Israel': 'yellow',
    'Palestine': 'pink',
    'France': 'cyan',
    'Germany': 'magenta',
    'Pakistan': 'brown',
    'Taiwan': 'lime'
}

# Initialize a Folium map with global coordinates
m = folium.Map(location=[20, 0], zoom_start=2, control_scale=True, tiles=None)

# Add a custom tile layer for ocean and land coloring
folium.TileLayer(
    tiles='Stamen Toner',  # You can also try 'CartoDB positron' or 'Stamen Watercolor'
    name='Toner',
    control=False
).add_to(m)

# Add country boundaries (GeoJson layer)
geo_json_url = 'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/world-countries.json'
folium.GeoJson(
    geo_json_url,
    style_function=lambda feature: {
        'fillColor': 'grey',  # Land color
        'color': 'black',  # Country borders color
        'weight': 2,  # Thicker borders
        'fillOpacity': 0.6,
    },
    tooltip=GeoJsonTooltip(
        fields=['name'],
        aliases=['Country:'],
        localize=True,
        sticky=True,
        labels=True,
        style=(
            "background-color: white; color: black; font-family: "
            "arial; font-size: 12px; padding: 10px;"
        )
    )
).add_to(m)

# Add circle markers for each country
for country, coords in geo_coords.items():
    folium.CircleMarker(
        location=coords,
        radius=10,
        color=country_colors[country],
        fill=True,
        fill_opacity=0.8,
        popup=f"<b>{country}</b>: {df[df['Country'] == country]['Conflict_Type'].values[0]}",
        tooltip=folium.Tooltip(f"<span style='color:black;'><b>{country}</b></span>")  # Country name in black when hovered
    ).add_to(m)



# Save the map to an HTML file and display
m.save('Global_Conflict_Map_with_UN_Involvement.html')
m


# In[ ]:





# In[14]:


import pandas as pd
import folium

# Sample data
data = {
    'Country': ['USA', 'Russia', 'India', 'China', 'Ukraine', 
                'Israel', 'Palestine', 'France', 'Germany', 
                'Pakistan', 'Taiwan'],
    'Conflict_Type': ['Tension', 'War', 'Tension', 'Potential Conflict', 
                      'War', 'War', 'War', 'Potential Conflict', 
                      'Tension', 'Extreme Tension', 'Potential Conflict'],
}

# Create a DataFrame
df = pd.DataFrame(data)

# Define geographical coordinates for countries
geo_coords = {
    'USA': [37.0902, -95.7129],
    'Russia': [61.5240, 105.3188],
    'India': [20.5937, 78.9629],
    'China': [35.8617, 104.1954],
    'Ukraine': [48.3794, 31.1656],
    'Israel': [31.0461, 34.8516],
    'Palestine': [31.9522, 35.2332],
    'France': [46.6034, 1.8883],
    'Germany': [51.1657, 10.4515],
    'Pakistan': [30.3753, 69.3451],
    'Taiwan': [23.6978, 120.9605]
}

# Define colors for circles based on countries
circle_colors = {
    'USA': 'blue',
    'Russia': 'red',
    'India': 'green',
    'China': 'yellow',
    'Ukraine': 'purple',
    'Israel': 'orange',
    'Palestine': 'pink',
    'France': 'cyan',
    'Germany': 'magenta',
    'Pakistan': 'lime',
    'Taiwan': 'teal'
}

# Initialize Folium map with a larger size and specific location
m = folium.Map(location=[20, 0], zoom_start=2, control_scale=True, 
                tiles='CartoDB positron')

# Add GeoJson layer for countries without borders (same color as land)
folium.GeoJson(
    'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/world-countries.json',
    style_function=lambda feature: {
        'fillColor': 'grey',   # Color of land
        'color': 'grey',       # Make outline color the same as land color
        'weight': 0,           # No weight for the outline to effectively remove it
        'fillOpacity': 0.6,
    }
).add_to(m)

# Add circles and country names on the map
for index, row in df.iterrows():
    country = row['Country']
    coords = geo_coords[country]
    
    # Create a circle marker for each country
    folium.CircleMarker(
        location=coords,
        radius=10,
        color=circle_colors[country],
        fill=True,
        fill_opacity=0.6,
        popup=f"<b>{country}</b>: {row['Conflict_Type']}",
    ).add_to(m)

# Display the map
m.save('UN_Interventions_Choropleth_Map.html')
m


# In[ ]:





# # Data Preprocessing for Machine Learning

# In[32]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Assuming df is already defined and loaded with the data

# One-hot encode categorical columns
X = pd.get_dummies(df[['Country', 'Conflict_Region', 'UN_Interventions']])

# Target column
y = df['Conflict_Type']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the numerical features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Assuming df is already defined and loaded with the data

# One-hot encode categorical columns
X = pd.get_dummies(df[['Country', 'Conflict_Region', 'UN_Interventions']])

# Target column
y = df['Conflict_Type']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the numerical features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("X_train_scaled : ", X_train_scaled)
print("\n\n\nX_test_scaled : ", X_test_scaled)


# #  Machine Learning Models for Conflict Prediction : Train-Test Split and Random Forest Model
# 

# In[33]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Prepare data
X = df[['Deaths', 'Economic_Impact_Billion', 'Environmental_Damage_Index']]
y = df['Conflict_Type']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Random Forest Classifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print(classification_report(y_test, y_pred))


# # Feature Engineering for Machine Learning Models

# In[41]:


df['Conflict_Binary'] = df['Conflict_Region'].apply(lambda x: 1 if x == 'Yes' else 0)
df['UN_Interventions'] = df['UN_Interventions'].apply(lambda x: 1 if x == 'Yes' else 0)


# In[32]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Sample dataset creation (this would be your actual dataset)
data = {
    'Country': ['USA', 'Russia', 'India', 'China', 'Ukraine', 
                'Israel', 'Palestine', 'France', 'Germany', 
                'Pakistan', 'Taiwan'],
    'Conflict_Type': ['Tension', 'War', 'Tension', 'Potential Conflict', 
                      'War', 'War', 'War', 'Potential Conflict', 
                      'Tension', 'Extreme Tension', 'Potential Conflict'],
    'Environmental_Damage_Index': np.random.uniform(1, 10, size=11),  # Adjusted to 11 entries
    'UN_Interventions': np.random.choice([1, 2, 3, 4], size=11),  # Adjusted to 11 entries
    'Conflict_Intensity': [10, 20, 30, 25, 15, 18, 5, 12, 22, 16, 14],
}

# Create a DataFrame
df = pd.DataFrame(data)

# 1. Handling Missing Values
# Filling missing values with median (assuming numerical columns)
df['Environmental_Damage_Index'] = df['Environmental_Damage_Index'].fillna(df['Environmental_Damage_Index'].median())

# 2. Encoding Categorical Variables
df = pd.get_dummies(df, columns=['Conflict_Type'], drop_first=True)

# 3. Feature Scaling (Normalization)
scaler = MinMaxScaler()
df['Environmental_Damage_Index'] = scaler.fit_transform(df[['Environmental_Damage_Index']])

# 4. Creating a New Feature: Conflict Intensity (example based on existing data)
df['Conflict_Intensity'] = df['Environmental_Damage_Index'] * df['UN_Interventions']

# 5. Display the processed DataFrame
print("Processed DataFrame:")
print(df)

# 6. Data Visualization: Distribution of Conflict Types
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Conflict_Type_War', palette='viridis', hue='Conflict_Type_War', legend=False)
plt.title('Distribution of Conflict Types')
plt.xlabel('Conflict Type (War = 1, Others = 0)')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['No War', 'War'])
plt.grid(axis='y')
plt.show()


# # Model Selection - Logistic Regression

# In[42]:


from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train)


# In[35]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Sample dataset creation using provided data
data = {
    'Country': ['USA', 'Russia', 'India', 'China', 'Ukraine', 'Israel', 'Palestine', 
                'France', 'Germany', 'Pakistan', 'Taiwan'],
    'Conflict_Region': ['Mid North America', 'Eastern Europe', 'South Asia', 'East Asia', 
                        'Eastern Europe', 'Middle East', 'Middle East', 'Western Europe', 
                        'Western Europe', 'South West Asia', 'East Asia'],
    'Conflict_Type': ['Tension', 'War', 'Tension', 'Potential Conflict', 'War', 
                      'War', 'War', 'Potential Conflict', 'Tension', 'Extreme Tension', 
                      'Potential Conflict'],
    'Latitude': [37.0902, 61.5240, 20.5937, 35.8617, 48.3794, 
                 31.0461, 31.9522, 46.6034, 51.1657, 30.3753, 23.6978],
    'Longitude': [-95.7129, 105.3188, 78.9629, 104.1954, 31.1656, 
                  34.8516, 35.2332, 1.8883, 10.4515, 69.3451, 121.0200],
    'Altitude': [760, 600, 160, 1840, 175, 508, 795, 375, 263, 900, 1150],
    'Conflict_Intensity': [10, 20, 30, 25, 15, 18, 5, 12, 22, 16, 14],
    'Deaths': np.random.randint(1000, 50000, size=11),
    'Economic_Impact_Billion': np.random.uniform(1.5, 100, size=11),
    'Environmental_Damage_Index': np.random.uniform(1, 10, size=11),
    'UN_Interventions': np.random.choice([1, 2, 3, 4], size=11),
    'Total_Population': [331002651, 145912025, 1380004385, 1439323776, 43733762, 
                         8655535, 5000000, 65273511, 83783942, 225199937, 23816775],
    'Male_Population': [162000000, 67000000, 705000000, 724000000, 22000000, 
                       4300000, 2500000, 32000000, 41000000, 113000000, 12000000],
    'Female_Population': [169000000, 78900000, 675000000, 715000000, 21700000, 
                         4350000, 2500000, 33200000, 42700000, 112000000, 11800000]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Encoding target variable as binary (1 for 'War', 0 for other types)
df['Conflict_Type'] = df['Conflict_Type'].apply(lambda x: 1 if x == 'War' else 0)

# Defining features and target variable
X = df[['Latitude', 'Longitude', 'Altitude', 'Conflict_Intensity', 
         'Deaths', 'Economic_Impact_Billion', 'Environmental_Damage_Index', 
         'UN_Interventions', 'Total_Population']]
y = df['Conflict_Type']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Creating and training the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Displaying evaluation results
print(f"Accuracy of the Logistic Regression model: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Visualization of the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No War', 'War'], yticklabels=['No War', 'War'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# #  Evaluate Logistic Regression Model

# In[43]:


from sklearn.metrics import classification_report

y_pred_log = log_reg.predict(X_test_scaled)
print(classification_report(y_test, y_pred_log))


# In[34]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc

# Sample dataset creation
data = {
    'Country': ['USA', 'Russia', 'India', 'China', 'Ukraine', 'Israel', 'Palestine', 
                'France', 'Germany', 'Pakistan', 'Taiwan'],
    'Conflict_Region': ['Mid North America', 'Eastern Europe', 'South Asia', 'East Asia', 
                        'Eastern Europe', 'Middle East', 'Middle East', 'Western Europe', 
                        'Western Europe', 'South West Asia', 'East Asia'],
    'Conflict_Type': ['Tension', 'War', 'Tension', 'Potential Conflict', 'War', 
                      'War', 'War', 'Potential Conflict', 'Tension', 'Extreme Tension', 
                      'Potential Conflict'],
    'Latitude': [37.0902, 61.5240, 20.5937, 35.8617, 48.3794, 31.0461, 31.9522, 
                 46.6034, 51.1657, 30.3753, 23.6978],
    'Longitude': [-95.7129, 105.3188, 78.9629, 104.1954, 31.1656, 34.8516, 35.2332, 
                  1.8883, 10.4515, 69.3451, 121.0200],
    'Altitude': [760, 600, 160, 1840, 175, 508, 795, 375, 263, 900, 1150],
    'Conflict_Intensity': [10, 20, 30, 25, 15, 18, 5, 12, 22, 16, 14],
    'Deaths': np.random.randint(1000, 50000, size=11),
    'Economic_Impact_Billion': np.random.uniform(1.5, 100, size=11),
    'Environmental_Damage_Index': np.random.uniform(1, 10, size=11),
    'UN_Interventions': np.random.choice([1, 2, 3, 4], size=11),
    'Total_Population': [331002651, 145912025, 1380004385, 1439323776, 43733762, 
                         8655535, 5000000, 65273511, 83783942, 225199937, 23816775],
    'Male_Population': [162000000, 67000000, 705000000, 724000000, 22000000, 
                       4300000, 2500000, 32000000, 41000000, 113000000, 12000000],
    'Female_Population': [169000000, 78900000, 675000000, 715000000, 21700000, 
                         4350000, 2500000, 33200000, 42700000, 112000000, 11800000]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Encoding target variable as binary (1 for 'War', 0 for other types)
df['Conflict_Type'] = df['Conflict_Type'].apply(lambda x: 1 if x == 'War' else 0)

# Defining features and target variable
X = df[['Latitude', 'Longitude', 'Altitude', 'Conflict_Intensity', 'Deaths', 
         'Economic_Impact_Billion', 'Environmental_Damage_Index', 'UN_Interventions']]
y = df['Conflict_Type']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Creating and training the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Displaying evaluation results
print(f"Accuracy of the Logistic Regression model: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

# Plotting the ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()


# # Hyperparameter Tuning for Logistic Regression

# In[2]:


from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# Example dataset (replace this with your actual dataset)
data = {'Feature1': [1, 2, 3, 4, 5, 6, 7, 8],
        'Feature2': [5, 6, 7, 8, 9, 10, 11, 12],
        'Target': [0, 1, 0, 1, 0, 1, 0, 1]}  # Make sure you have enough samples per class

df = pd.DataFrame(data)

# Features and target
X = df[['Feature1', 'Feature2']]
y = df['Target']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize logistic regression model
log_reg = LogisticRegression()

# Define parameter grid
param_grid = {'C': [0.1, 1, 10, 100]}

# Use StratifiedKFold for better handling of imbalanced classes (if applicable)
# Make sure the number of splits is smaller than or equal to the number of samples in the smallest class
cv = StratifiedKFold(n_splits=2)  # Adjust n_splits based on your dataset

# Grid search with cross-validation
grid_search = GridSearchCV(log_reg, param_grid, cv=cv)
grid_search.fit(X_train_scaled, y_train)

# Print the best parameters
print(grid_search.best_params_)


# # Model Selection - Random Forest Classifier

# In[3]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)


# # Evaluate Random Forest Classifier

# In[5]:


# Import the necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Example dataset (replace with your actual dataset)
data = {'Feature1': [1, 2, 3, 4, 5, 6, 7, 8],
        'Feature2': [5, 6, 7, 8, 9, 10, 11, 12],
        'Target': [0, 1, 0, 1, 0, 1, 0, 1]}  # Ensure there are sufficient samples for each class

df = pd.DataFrame(data)

# Features and target
X = df[['Feature1', 'Feature2']]
y = df['Target']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest Classifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_scaled, y_train)

# Make predictions
y_pred_rf = rf.predict(X_test_scaled)

# Print the classification report
print(classification_report(y_test, y_pred_rf))


# # Feature Importance for Random Forest

# In[6]:


import matplotlib.pyplot as plt
import seaborn as sns

feature_importances = rf.feature_importances_
sns.barplot(x=feature_importances, y=X.columns)
plt.title('Feature Importance in Random Forest')
plt.show()


# # Deep Learning Model - Neural Network

# In[7]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32)


# # Evaluate Neural Network Model

# In[1]:


# Import necessary libraries
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Sample dataset for model training (replace with real data)
# For simplicity, we're creating random data for X and y
np.random.seed(42)
X = np.random.rand(100, 10)  # 100 samples, 10 features
y = np.random.randint(2, size=100)  # Binary target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the model
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=10, batch_size=10, verbose=0)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f'Test Accuracy: {test_acc:.4f}')

# Create a Violin Plot for conflict severity across regions
# Sample dataset for the violin plot
data = {
    'region': ['Africa', 'Asia', 'Europe', 'America', 'Africa', 'Asia', 'Europe', 'America'],
    'conflict_severity': [8, 6, 5, 9, 7, 6, 4, 8]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Violin plot for conflict severity across regions
plt.figure(figsize=(10, 6))
sns.violinplot(x='region', y='conflict_severity', data=df, palette='Set2')

# Add labels and title
plt.title('Distribution of Conflict Severity Across Regions', fontsize=16)
plt.xlabel('Region', fontsize=12)
plt.ylabel('Conflict Severity', fontsize=12)

# Show the plot
plt.show()


# # Predictive Modeling using Time Series Analysis

# In[10]:


# Import necessary libraries
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Create a simple time series data (example)
# In practice, you can load data from a CSV or another source.
dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
values = [i + (i * 0.5) for i in range(100)]  # Example linear trend
time_series_data = pd.Series(values, index=dates)

# Visualize the time series
plt.figure(figsize=(10, 6))
time_series_data.plot(title="Sample Time Series Data")
plt.show()

# Fit the ARIMA model
model = ARIMA(time_series_data, order=(1, 1, 1))  # Example ARIMA(1,1,1) model
model_fit = model.fit()

# Print the summary of the ARIMA model
print(model_fit.summary())


# # NLP - Sentiment Analysis on UN Documents

# In[2]:


# Import necessary libraries
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize the Sentiment Intensity Analyzer
sid = SentimentIntensityAnalyzer()

# Sample UN resolution text (replace with actual UN document text)
text = "The United Nations seeks to promote global peace and security by addressing conflicts through diplomacy and international law."

# Perform Sentiment Analysis
sentiment_scores = sid.polarity_scores(text)
print("Sentiment Scores:", sentiment_scores)

# Data Visualization - Bar Chart of Sentiment Scores
# Convert sentiment scores into two lists: keys and values
labels = list(sentiment_scores.keys())
scores = list(sentiment_scores.values())

# Create a bar chart for sentiment analysis
plt.figure(figsize=(8, 6))
sns.barplot(x=labels, y=scores, palette='Blues_d')

# Add title and labels
plt.title('Sentiment Analysis of UN Resolution Text', fontsize=16)
plt.xlabel('Sentiment Categories', fontsize=12)
plt.ylabel('Scores', fontsize=12)

# Show the plot
plt.show()


# # 3D Geographical Visualization using Plotly and Basemap and Geopandas

# In[13]:


import folium
import pandas as pd

# Data for the countries with altitude
data = {
    'Country': [
        'USA', 'Russia', 'India', 'China', 'Ukraine', 
        'Israel', 'Palestine', 'France', 'Germany', 'Pakistan', 'Taiwan'
    ],
    'Latitude': [
        37.0902, 61.5240, 20.5937, 35.8617, 48.3794, 
        31.0461, 31.9522, 46.6034, 51.1657, 30.3753, 23.6978
    ],
    'Longitude': [
        -95.7129, 105.3188, 78.9629, 104.1954, 31.1656, 
        34.8516, 35.2332, 1.8883, 10.4515, 69.3451, 121.0200
    ],
    'Altitude': [
        760, 600, 160, 1840, 175, 
        508, 795, 375, 263, 900, 1150  # Actual average altitudes in meters
    ],
    'Conflict_Intensity': [
        10, 20, 30, 25, 15, 18, 5, 12, 22, 16, 14  # Example conflict intensity values
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Create a map object centered around the geographic mid-point
m = folium.Map(location=[20, 0], zoom_start=2)

# Add markers for each country
for i, row in df.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=(
            f"<b>Country:</b> {row['Country']}<br>"
            f"<b>Altitude:</b> {row['Altitude']} meters<br>"
            f"<b>Conflict Intensity:</b> {row['Conflict_Intensity']}"
        ),
        tooltip=row['Country'],
        icon=folium.Icon(color='blue' if row['Conflict_Intensity'] < 15 else 'red')
    ).add_to(m)

# Save map to an HTML file and display it
m.save('3d_map_countries.html')
m


# In[41]:


pip install basemap


# # 2D Spherical Globe

# In[45]:


import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

def draw_globe():
    # Create a new figure
    fig = plt.figure(figsize=(10, 10))

    # Create a Basemap instance for a globe projection
    m = Basemap(projection='ortho', lat_0=0, lon_0=0)

    # Draw coastlines, countries, and the edges of the map
    m.drawcoastlines()
    m.drawcountries()
    m.drawmapboundary(fill_color='aqua')  # Color for oceans
    m.fillcontinents(color='lightgreen', lake_color='aqua')

    # Add a title
    plt.title('Spherical Movable Globe', fontsize=20)

    # Save the figure to the specified path
    plt.savefig("E:\\spherical_globe.png", bbox_inches='tight', dpi=300)

    # Display the globe
    plt.show()

# Call the function to draw the globe
draw_globe()


# # 3D Spherical Globe

# In[2]:


import plotly.graph_objects as go
import pandas as pd

# Load country data
url = "https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv"
df = pd.read_csv(url)

# Inspect the DataFrame to check the column names
print(df.head())  # Print the first few rows of the DataFrame for verification

# Check the columns in the DataFrame
print("Columns in the DataFrame:", df.columns)

# Example mapping of countries to coordinates (for demonstration)
coordinates = {
    'United States': (-95.71, 37.09),
    'China': (104.19, 35.86),
    'India': (78.96, 20.59),
    'Brazil': (-51.93, -14.24),
    'Australia': (133.77, -25.27),
    # Add more countries and their coordinates as needed
}

# Create Longitude and Latitude columns using the mapping
df['Longitude'] = df['COUNTRY'].map(lambda x: coordinates.get(x, (None, None))[0])
df['Latitude'] = df['COUNTRY'].map(lambda x: coordinates.get(x, (None, None))[1])

# Create a scatter plot for the globe
fig = go.Figure()

# Add a globe surface
fig.add_trace(go.Scattergeo(
    lon=df['Longitude'],
    lat=df['Latitude'],
    text=df['COUNTRY'],
    mode='markers',
    marker=dict(size=6, color='blue', opacity=0.7),
))

# Customize layout for the globe
fig.update_layout(
    title='Interactive 3D Earth',
    geo=dict(
        projection_type='orthographic',
        showland=True,
        landcolor='lightgreen',
        subunitcolor='white',
        countrycolor='white',
        showocean=True,
        oceancolor='aqua',
        showcountries=True,
    ),
    updatemenus=[
        {
            'buttons': [
                {
                    'label': 'Zoom In',
                    'method': 'relayout',
                    'args': ['geo.projection.scale', 1.2]  # Increase scale for zooming in
                },
                {
                    'label': 'Zoom Out',
                    'method': 'relayout',
                    'args': ['geo.projection.scale', 0.8]  # Decrease scale for zooming out
                }
            ],
            'direction': 'down',
            'showactive': True,
            'x': 0.1,
            'xanchor': 'left',
            'y': 1.1,
            'yanchor': 'top'
        }
    ],
    width=1400,  # Increase width to triple the size
    height=1400,  # Increase height to triple the size
)

# Function to rotate the globe to the specified country
def rotate_globe(country_name):
    # Get the coordinates of the specified country
    country_data = df[df['COUNTRY'] == country_name]
    
    if not country_data.empty:
        lon = country_data['Longitude'].values[0]
        lat = country_data['Latitude'].values[0]

        # Update camera position to focus on the specified country
        fig.update_layout(
            geo=dict(
                projection_type='orthographic',
                center=dict(lat=lat, lon=lon),
            )
        )
        fig.show()
    else:
        print("Country not found. Please enter a valid country name.")

# Display the globe
fig.show()


# In[3]:


import plotly.graph_objects as go
import pandas as pd
from geopy.geocoders import Nominatim
import time

# Load country data
url = "https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv"
df = pd.read_csv(url)

# Inspect the DataFrame to check the column names
print(df.head())  # Print the first few rows of the DataFrame for verification

# Check the columns in the DataFrame
print("Columns in the DataFrame:", df.columns)

# Initialize geocoder
geolocator = Nominatim(user_agent="geoapiExercises")

# Function to get coordinates for a country
def get_coordinates(country_name):
    try:
        location = geolocator.geocode(country_name)
        return (location.longitude, location.latitude) if location else (None, None)
    except Exception as e:
        print(f"Error getting coordinates for {country_name}: {e}")
        return (None, None)

# Get coordinates for each country in the DataFrame
coordinates = df['COUNTRY'].apply(get_coordinates)
df['Longitude'] = coordinates.apply(lambda x: x[0])
df['Latitude'] = coordinates.apply(lambda x: x[1])

# Add a small delay to avoid overwhelming the geocoding service
time.sleep(1)

# Create a scatter plot for the globe
fig = go.Figure()

# Add a globe surface with country names
fig.add_trace(go.Scattergeo(
    lon=df['Longitude'],
    lat=df['Latitude'],
    text=df['COUNTRY'],
    mode='text',  # Use 'text' mode to display country names
    textfont=dict(size=8, color='blue'),
))

# Customize layout for the globe
fig.update_layout(
    title='Interactive 3D Earth with Country Names',
    geo=dict(
        projection_type='orthographic',
        showland=True,
        landcolor='lightgreen',
        subunitcolor='white',
        countrycolor='white',
        showocean=True,
        oceancolor='aqua',
        showcountries=True,
    ),
    updatemenus=[
        {
            'buttons': [
                {
                    'label': 'Zoom In',
                    'method': 'relayout',
                    'args': ['geo.projection.scale', 1.2]  # Increase scale for zooming in
                },
                {
                    'label': 'Zoom Out',
                    'method': 'relayout',
                    'args': ['geo.projection.scale', 0.8]  # Decrease scale for zooming out
                }
            ],
            'direction': 'down',
            'showactive': True,
            'x': 0.1,
            'xanchor': 'left',
            'y': 1.1,
            'yanchor': 'top'
        }
    ],
    width=1400,  # Increase width to triple the size
    height=1400,  # Increase height to triple the size
)

# Function to rotate the globe to the specified country
def rotate_globe(country_name):
    # Get the coordinates of the specified country
    country_data = df[df['COUNTRY'] == country_name]
    
    if not country_data.empty:
        lon = country_data['Longitude'].values[0]
        lat = country_data['Latitude'].values[0]

        # Update camera position to focus on the specified country
        fig.update_layout(
            geo=dict(
                projection_type='orthographic',
                center=dict(lat=lat, lon=lon),
            )
        )
        fig.show()
    else:
        print("Country not found. Please enter a valid country name.")

# Display the globe
fig.show()


# In[ ]:





# #  Choropleth Map Visualization

# In[12]:


import folium
import pandas as pd
import json
import requests

# URL for the GeoJSON file
geojson_url = 'https://raw.githubusercontent.com/datasets/geo-boundaries-world-110m/master/countries.geojson'

# Download the GeoJSON file
response = requests.get(geojson_url)
with open('E:/world_countries.geojson', 'wb') as f:
    f.write(response.content)

# Load conflict data - Example DataFrame
# This should be replaced with your actual conflict data loading
data = {
    'Country': ['USA', 'India', 'China', 'Russia', 'Israel'],
    'UN_Interventions': [1, 3, 2, 2, 2]  # Numeric values for conflict intensity
}
conflict_data = pd.DataFrame(data)

# Load geo data from the downloaded GeoJSON file
with open('E:/world_countries.geojson') as f:
    geo_data = json.load(f)

# Create a map centered at a given location
world_map = folium.Map(location=[20, 0], zoom_start=2)

# Create a choropleth map
folium.Choropleth(
    geo_data=geo_data,
    data=conflict_data,
    columns=['Country', 'UN_Interventions'],  # Ensure these columns exist in conflict_data
    key_on='feature.properties.name',  # Adjust key_on based on your geo_data structure
    fill_color='YlOrRd',
    legend_name='UN_Interventions'  # Updated legend name
).add_to(world_map)

# Display the map
world_map


# # Bubble Map Visualization

# In[6]:


import plotly.express as px

# Sample DataFrame (replace 'df' with your actual DataFrame containing 'Country' and 'Conflict_Intensity')
data = {
    'Country': ['India', 'United States', 'China', 'Russia', 'Germany'],
    'Conflict_Intensity': [70, 30, 50, 40, 25]
}

import pandas as pd
df = pd.DataFrame(data)

# Create a scatter_geo plot with bubble sizes based on conflict intensity
fig = px.scatter_geo(df, locations="Country", locationmode='country names', 
                     size="Conflict_Intensity", projection="natural earth", 
                     size_max=50)  # Increase or decrease the value to control bubble size

# Set the width and height of the bubble map
fig.update_layout(width=1000, height=800)  # Adjust these values as needed

# Show the plot
fig.show()


# # Clustering - KMeans for Conflict Classification

# In[3]:


# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# 1. Scaling the data using StandardScaler (you can use MinMaxScaler if needed)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Scale the features

# 2. Applying KMeans clustering
kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(X_scaled)

# 3. Adding the cluster labels to the DataFrame
df['Cluster'] = clusters

# Display the DataFrame with clusters
print(df)


# # Dimensionality Reduction using PCA

# In[8]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



# Assuming y_train is a simple target variable (e.g., binary classification)
y_train = np.array([0, 1, 0, 1, 0])

# Scaling the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(df)

# Applying PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

# Visualizing the PCA components
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap='viridis')  # Color by target (y_train)
plt.title("PCA on Conflict Data")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()
plt.show()


# # Anomaly Detection using Isolation Forest

# In[7]:


import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data for conflict mapping
data = {
    'Country': ['USA', 'Russia', 'India', 'China', 'Ukraine', 'Israel', 'Palestine', 'France', 'Germany', 'Pakistan', 'Taiwan'],
    'Conflict_Region': ['Mid North America', 'Eastern Europe', 'South Asia', 'East Asia', 'Eastern Europe', 'Middle East', 'Middle East', 'Western Europe', 'Western Europe', 'South West Asia', 'East Asia'],
    'Conflict_Type': ['Tension', 'War', 'Tension', 'Potential Conflict', 'War', 'War', 'War', 'Potential Conflict', 'Tension', 'Extreme Tension', 'Potential Conflict'],
    'Latitude': [37.0902, 61.5240, 20.5937, 35.8617, 48.3794, 31.0461, 31.9522, 46.6034, 51.1657, 30.3753, 23.6978],
    'Longitude': [-95.7129, 105.3188, 78.9629, 104.1954, 31.1656, 34.8516, 35.2332, 1.8883, 10.4515, 69.3451, 121.0200],
    'Altitude': [760, 600, 160, 1840, 175, 508, 795, 375, 263, 900, 1150],
    'Conflict_Intensity': [10, 20, 30, 25, 15, 18, 5, 12, 22, 16, 14],
    'Deaths': np.random.randint(1000, 50000, size=11),
    'Economic_Impact_Billion': np.random.uniform(1.5, 100, size=11),
    'Environmental_Damage_Index': np.random.uniform(1, 10, size=11),
    'UN_Interventions': np.random.choice([1, 2, 3, 4], size=11),
    'Total_Population': [331002651, 145912025, 1380004385, 1439323776, 43733762, 8655535, 5000000, 65273511, 83783942, 225199937, 23816775],
    'Male_Population': [162000000, 67000000, 705000000, 724000000, 22000000, 4300000, 2500000, 32000000, 41000000, 113000000, 12000000],
    'Female_Population': [169000000, 78900000, 675000000, 715000000, 21700000, 4350000, 2500000, 33200000, 42700000, 112000000, 11800000]
}

df = pd.DataFrame(data)

# Features for anomaly detection
features = ['Conflict_Intensity', 'Deaths', 'Economic_Impact_Billion', 'Environmental_Damage_Index', 'UN_Interventions', 'Altitude']
X = df[features]

# Initialize and fit the Isolation Forest
isolation_forest = IsolationForest(contamination=0.1, random_state=42)
df['Anomaly'] = isolation_forest.fit_predict(X)

# Print anomalies
anomalies = df[df['Anomaly'] == -1]
print(f"Anomalies detected:\n{anomalies}")

# Create a Swarm Plot to visualize Conflict Intensity vs Region
plt.figure(figsize=(12, 6))
sns.swarmplot(x='Conflict_Region', y='Conflict_Intensity', data=df, hue='Anomaly', palette='Set2')

# Add labels and title
plt.title('Swarm Plot of Conflict Intensity by Region (with Anomalies)', fontsize=16)
plt.xlabel('Conflict Region', fontsize=12)
plt.ylabel('Conflict Intensity', fontsize=12)

# Show the plot
plt.show()


# # Gantt Chart for Conflict Timelines

# In[6]:


import plotly.figure_factory as ff

df = [dict(Task='Conflict 1', Start='2022-01-01', Finish='2022-12-31'),
      dict(Task='Conflict 2', Start='2021-05-15', Finish='2022-09-30')]
fig = ff.create_gantt(df)
fig.show()


# #  Feature Importance using XGBoost

# In[8]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris  # Using Iris dataset as an example
import xgboost as xgb
import matplotlib.pyplot as plt

# Load the dataset (you can replace this with your own dataset)
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target

# 1. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Initialize and train the XGBoost classifier
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)

# 3. Plot feature importance
xgb.plot_importance(xgb_model)
plt.show()


# In[4]:


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import make_classification
import pandas as pd

# Creating a synthetic dataset with 1000 samples and 20 features for binary classification
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                           n_redundant=5, n_classes=2, random_state=42)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Building the XGBoost Model
xgboost_model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, seed=42)
xgboost_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_xgb = xgboost_model.predict(X_test_scaled)

# Evaluating the model
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Classification Report:\n", classification_report(y_test, y_pred_xgb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))


# # Ternary Plot for Visualizing Multiclass Conflict Data

# In[6]:


import plotly.express as px
import pandas as pd

# Example: Create a DataFrame (replace this with your actual DataFrame)
data = {
    'India': [0.2, 0.3, 0.1, 0.4],
    'Russia': [0.3, 0.1, 0.4, 0.2],
    'China': [0.5, 0.6, 0.5, 0.4],
    'Conflict_Intensity': [10, 20, 15, 25]
}

df = pd.DataFrame(data)

# Verify column names (check if "India", "Russia", "Pakistan" exist)
print(df.columns)

# Create a scatter ternary plot
fig = px.scatter_ternary(df, a="India", b="Russia", c="China", size="Conflict_Intensity")
fig.show()


# # Sankey Diagram for Conflict Flow

# In[11]:


import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# Define the dataset
data = {
    'Country': ['USA', 'Russia', 'India', 'China', 'Ukraine', 'Israel', 'Palestine', 'France', 'Germany', 'Pakistan', 'Taiwan'],
    'Conflict_Region': ['Mid North America', 'Eastern Europe', 'South Asia', 'East Asia', 'Eastern Europe', 
                        'Middle East', 'Middle East', 'Western Europe', 'Western Europe', 'South West Asia', 'East Asia'],
    'Conflict_Type': ['Tension', 'War', 'Tension', 'Potential Conflict', 'War', 'War', 'War', 
                      'Potential Conflict', 'Tension', 'Extreme Tension', 'Potential Conflict'],
    'Latitude': [37.0902, 61.5240, 20.5937, 35.8617, 48.3794, 31.0461, 31.9522, 46.6034, 51.1657, 30.3753, 23.6978],
    'Longitude': [-95.7129, 105.3188, 78.9629, 104.1954, 31.1656, 34.8516, 35.2332, 1.8883, 10.4515, 69.3451, 121.0200],
    'Altitude': [760, 600, 160, 1840, 175, 508, 795, 375, 263, 900, 1150],
    'Conflict_Intensity': [10, 20, 30, 25, 15, 18, 5, 12, 22, 16, 14],
    'Deaths': np.random.randint(1000, 50000, size=11),
    'Economic_Impact_Billion': np.random.uniform(1.5, 100, size=11),
    'Environmental_Damage_Index': np.random.uniform(1, 10, size=11),
    'UN_Interventions': np.random.choice([1, 2, 3, 4], size=11),
    'Total_Population': [331002651, 145912025, 1380004385, 1439323776, 43733762, 8655535, 5000000, 
                         65273511, 83783942, 225199937, 23816775],
    'Male_Population': [162000000, 67000000, 705000000, 724000000, 22000000, 4300000, 2500000, 
                       32000000, 41000000, 113000000, 12000000],
    'Female_Population': [169000000, 78900000, 675000000, 715000000, 21700000, 4350000, 2500000, 
                         33200000, 42700000, 112000000, 11800000]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Create a source-target mapping for Sankey Diagram
sources = df['Conflict_Type']
targets = df['Conflict_Region']
values = df['Conflict_Intensity']

# Create unique labels for the Sankey diagram
all_labels = pd.concat([sources, targets]).unique()
source_indices = [list(all_labels).index(s) for s in sources]
target_indices = [list(all_labels).index(t) + len(sources) for t in targets]

# Initialize the Sankey Diagram
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color='black', width=0.5),
        label=list(all_labels),
        color='blue'
    ),
    link=dict(
        source=source_indices,
        target=target_indices,
        value=values
    )
)])

# Update layout with custom size
fig.update_layout(
    title_text="Sankey Diagram for Conflict Flow",
    font_size=20,
    width=1000,  # Adjust width as needed
    height=600   # Adjust height as needed
)

fig.show()


# # Heatmap for Correlation Analysis

# In[4]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Sample data to create a DataFrame
data = {
    'Feature1': [1, 2, 3, 4, 5],
    'Feature2': [5, 4, 3, 2, 1],
    'Feature3': [2, 3, 4, 5, 6],
    'Feature4': [10, 9, 8, 7, 6]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")  # Creating the heatmap
plt.title('Heatmap of Feature Correlations')  # Adding a title
plt.show()  # Display the plot


# # Radar Chart for Conflict Severity

# In[11]:


from math import pi
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Example DataFrame structure (replace this with actual data)
data = {
    'Country': ['USA', 'Russia', 'India', 'China', 'Ukraine', 'Israel', 'Palestine', 'France', 'Germany', 'Pakistan', 'Taiwan'],
    'Conflict_Region': ['Mid North America', 'Eastern Europe', 'South Asia', 'East Asia', 'Eastern Europe', 'Middle East', 'Middle East', 'Western Europe', 'Western Europe', 'South West Asia', 'East Asia'],
    'Conflict_Type': ['Tension', 'War', 'Tension', 'Potential Conflict', 'War', 'War', 'War', 'Potential Conflict', 'Tension', 'Extreme Tension', 'Potential Conflict'],
    'Latitude': [37.0902, 61.5240, 20.5937, 35.8617, 48.3794, 31.0461, 31.9522, 46.6034, 51.1657, 30.3753, 23.6978],
    'Longitude': [-95.7129, 105.3188, 78.9629, 104.1954, 31.1656, 34.8516, 35.2332, 1.8883, 10.4515, 69.3451, 121.0200],
    'Altitude': [760, 600, 160, 1840, 175, 508, 795, 375, 263, 900, 1150],  # Corrected closing bracket for altitude
}
df = pd.DataFrame(data)

# Encoding the categorical columns 'Conflict_Region' and 'Conflict_Type'
label_encoder = LabelEncoder()
df['Conflict_Region'] = label_encoder.fit_transform(df['Conflict_Region'])
df['Conflict_Type'] = label_encoder.fit_transform(df['Conflict_Type'])

# Now using the correct column names
categories = ['Conflict_Region', 'Conflict_Type', 'Latitude', 'Longitude']  # No 'Country' for mean calculation

# Preparing the data for radar chart
conflict_data = df[categories].mean().values

# Number of variables
N = len(categories)

# Radar chart angles
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

conflict_data = np.concatenate([conflict_data, [conflict_data[0]]])

plt.figure(figsize=(6, 6))
ax = plt.subplot(111, polar=True)
plt.xticks(angles[:-1], categories)

# Plot data
ax.plot(angles, conflict_data, linewidth=2, linestyle='solid')
ax.fill(angles, conflict_data, 'b', alpha=0.4)

plt.title("Radar Chart for Conflict Severity")
plt.show()


# # WordCloud for Textual Analysis

# In[7]:


import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Sample DataFrame with exact conflict types and related UN report texts
data = {'Conflict_Type': ['Syrian Civil War', 'Yemeni Civil War', 'Sudan Conflict', 'Rohingya Crisis', 
                          'Libyan Civil War', 'Afghanistan Conflict', 'Israeli–Palestinian Conflict', 
                          'Ukraine Conflict', 'Somali Civil War', 'Ethiopian Civil Conflict', 
                          'Kashmir Conflict']}

df = pd.DataFrame(data)

# Actual brief UN report texts related to each conflict (replace with accurate summaries)
un_reports = [
    "The Syrian conflict has resulted in over 13 million people in need of humanitarian assistance.",
    "The Yemeni conflict has led to the world’s worst humanitarian crisis, with millions facing famine.",
    "The Sudan conflict has displaced millions, causing severe food insecurity and humanitarian concerns.",
    "UN officials report that the Rohingya crisis has displaced hundreds of thousands from Myanmar.",
    "Libya remains in turmoil post-Gaddafi, with rival factions and international involvement prolonging the conflict.",
    "Years of fighting in Afghanistan have left millions displaced and thousands of civilian casualties.",
    "The Israeli-Palestinian conflict continues to result in high casualties and deepening humanitarian crises.",
    "The Ukraine conflict has caused thousands of deaths and displaced millions, creating a large refugee crisis.",
    "Somalia remains embroiled in conflict, with the UN reporting widespread hunger and displacement.",
    "The Ethiopian civil conflict has led to mass displacement and a deepening humanitarian emergency.",
    "The Kashmir conflict between India and Pakistan continues to cause tensions, with significant human rights concerns."
]

# Ensure the UN report texts match the number of rows in the DataFrame
df['UN_report'] = un_reports

# Now, generate the WordCloud
text = " ".join(str(report) for report in df['UN_report'])

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("WordCloud of UN Reports on Global Conflicts")
plt.show()


# # Time Series Forecasting with ARIMA

# In[22]:


import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Sample DataFrame (adjust this to your actual data)
data = {
    'Country': ['USA', 'Russia', 'India', 'China', 'Ukraine', 'Israel', 'Palestine', 'France', 'Germany', 'Pakistan', 'Taiwan'],
    'conflict_intensity': [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]  # Sample conflict intensity data
}
df = pd.DataFrame(data)

# Fitting an ARIMA model to forecast conflict intensity over time
conflict_time_series = df.set_index('Country')['conflict_intensity']
arima_model = sm.tsa.ARIMA(conflict_time_series, order=(5, 1, 0))
arima_results = arima_model.fit()

# Forecasting
forecast = arima_results.forecast(steps=10)

# Adjust the plot size here
plt.figure(figsize=(18, 6))  # Change width (12) and height (6) as needed
plt.plot(conflict_time_series, label="Observed")
plt.plot(forecast, label="Forecast", color="red")
plt.title("ARIMA Forecast of Conflict Intensity")
plt.legend()
plt.show()


# #  Sunburst Chart for Conflict Types

# In[46]:


import pandas as pd
import numpy as np
import plotly.express as px  # Importing plotly express as px

# Sample DataFrame (adjust this to your actual data)
data = {
    'Country': ['USA', 'Russia', 'India', 'China', 'Ukraine', 'Israel', 'Palestine', 'France', 'Germany', 'Pakistan', 'Taiwan'],
    'Conflict_Region': ['Mid North America', 'Eastern Europe', 'South Asia', 'East Asia', 'Eastern Europe', 'Middle East', 'Middle East', 'Western Europe', 'Western Europe', 'South West Asia', 'East Asia'],
    'Conflict_Type': ['Tension', 'War', 'Tension', 'Potential Conflict', 'War', 'War', 'War', 'Potential Conflict', 'Tension', 'Extreme Tension', 'Potential Conflict'],
    'Latitude': [37.0902, 61.5240, 20.5937, 35.8617, 48.3794, 31.0461, 31.9522, 46.6034, 51.1657, 30.3753, 23.6978],
    'Longitude': [-95.7129, 105.3188, 78.9629, 104.1954, 31.1656, 34.8516, 35.2332, 1.8883, 10.4515, 69.3451, 121.0200],
    'Altitude': [760, 600, 160, 1840, 175, 508, 795, 375, 263, 900, 1150],
    'Conflict_Intensity': [10, 20, 30, 25, 15, 18, 5, 12, 22, 16, 14],
    'Deaths': np.random.randint(1000, 50000, size=11),  # Adjusted to 11 entries
    'Economic_Impact_Billion': np.random.uniform(1.5, 100, size=11),  # Adjusted to 11 entries
    'Environmental_Damage_Index': np.random.uniform(1, 10, size=11),  # Adjusted to 11 entries
    'UN_Interventions': np.random.choice([1, 2, 3, 4], size=11),  # Adjusted to 11 entries
    'Total_Population': [331002651, 145912025, 1380004385, 1439323776, 43733762, 8655535, 5000000, 65273511, 83783942, 225199937, 23816775],
    'Male_Population': [162000000, 67000000, 705000000, 724000000, 22000000, 4300000, 2500000, 32000000, 41000000, 113000000, 12000000],
    'Female_Population': [169000000, 78900000, 675000000, 715000000, 21700000, 4350000, 2500000, 33200000, 42700000, 112000000, 11800000]
}
df = pd.DataFrame(data)

# Sunburst chart to show hierarchical breakdown of conflict types by country and region
fig = px.sunburst(df, path=['Conflict_Region', 'Country', 'Conflict_Type'], values='Conflict_Intensity')  # Corrected column names

# Set the size of the chart
fig.update_layout(
    title="Sunburst Chart of Conflict Types by Region and Country",
    width=800,  # Set desired width in pixels
    height=600  # Set desired height in pixels
)

fig.show()


# # Sankey Diagram for Conflict Flows

# In[26]:


import plotly.graph_objects as go

# Sankey Diagram representing conflict flows between countries
fig = go.Figure(go.Sankey(
    node = dict(
      label = ["USA", "Russia", "Ukraine", "UN", "EU"],
      pad = 15, thickness = 20
    ),
    link = dict(
      source = [0, 1, 1, 2],  # Indices of the source nodes
      target = [2, 2, 3, 4],  # Indices of the target nodes
      value = [8, 4, 2, 6]  # Value for each link
    )
))

fig.update_layout(title="Sankey Diagram of Conflict Flows")
fig.show()


# # Voronoi Diagram for Regional Conflict Intensity

# In[28]:


from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt  # Importing matplotlib for plotting

# Ensure that the points are correctly referenced using the right column names
points = df[['Latitude', 'Longitude']].values  # Use 'Latitude' and 'Longitude' with capital 'L'

# Create the Voronoi diagram
vor = Voronoi(points)

# Plot the Voronoi diagram
voronoi_plot_2d(vor)
plt.title("Voronoi Diagram of Regional Conflict Intensity")
plt.show()


# # Hexbin Map for Geopolitical Conflicts

# In[30]:


# Hexbin map showing density of conflicts
plt.figure(figsize=(10, 6))
plt.hexbin(df['Longitude'], df['Latitude'], gridsize=50, cmap='coolwarm', reduce_C_function=np.mean)
plt.colorbar(label="Conflict Intensity")
plt.title("Hexbin Map of Geopolitical Conflicts")
plt.show()


# # Contour Plot for Conflict Severity

# In[33]:


# Contour plot for visualizing conflict severity
plt.figure(figsize=(10, 6))
plt.tricontourf(df['Longitude'], df['Latitude'], df['Conflict_Intensity'], cmap="RdYlBu")
plt.colorbar(label="Conflict Severity")
plt.title("Contour Plot of Conflict Severity by Region")
plt.show()


# # Box Plot for Economic Impact of Conflicts

# In[36]:


import matplotlib.pyplot as plt
import seaborn as sns

# Box plot of economic impact of conflicts across different regions
plt.figure(figsize=(12, 8))  # Increase the size by adjusting the values (width, height)
sns.boxplot(x='Conflict_Region', y='Economic_Impact_Billion', data=df)
plt.title("Box Plot of Economic Impact of Conflicts by Region")
plt.xticks(rotation=45)  # Optional: Rotate x-axis labels for better visibility
plt.show()


# # Population Pyramid for Affected Areas

# In[14]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Sample Data
data = {
    'Country': ['USA', 'Russia', 'India', 'China', 'Ukraine', 'Israel', 'Palestine', 'France', 'Germany', 'Pakistan', 'Taiwan'],
    'Conflict_Region': ['Mid North America', 'Eastern Europe', 'South Asia', 'East Asia', 'Eastern Europe', 'Middle East', 'Middle East', 'Western Europe', 'Western Europe', 'South West Asia', 'East Asia'],
    'Conflict_Type': ['Tension', 'War', 'Tension', 'Potential Conflict', 'War', 'War', 'War', 'Potential Conflict', 'Tension', 'Extreme Tension', 'Potential Conflict'],
    'Latitude': [37.0902, 61.5240, 20.5937, 35.8617, 48.3794, 31.0461, 31.9522, 46.6034, 51.1657, 30.3753, 23.6978],
    'Longitude': [-95.7129, 105.3188, 78.9629, 104.1954, 31.1656, 34.8516, 35.2332, 1.8883, 10.4515, 69.3451, 121.0200],
    'Altitude': [760, 600, 160, 1840, 175, 508, 795, 375, 263, 900, 1150],
    'Conflict_Intensity': [10, 20, 30, 25, 15, 18, 5, 12, 22, 16, 14],
    'Deaths': np.random.randint(1000, 50000, size=11),  # Adjusted to 11 entries
    'Economic_Impact_Billion': np.random.uniform(1.5, 100, size=11),  # Adjusted to 11 entries
    'Environmental_Damage_Index': np.random.uniform(1, 10, size=11),  # Adjusted to 11 entries
    'UN_Interventions': np.random.choice([1, 2, 3, 4], size=11),  # Adjusted to 11 entries
    'Total_Population': [331002651, 145912025, 1380004385, 1439323776, 43733762, 8655535, 5000000, 65273511, 83783942, 225199937, 23816775],
    'Male_Population': [162000000, 67000000, 705000000, 724000000, 22000000, 4300000, 2500000, 32000000, 41000000, 113000000, 12000000],
    'Female_Population': [169000000, 78900000, 675000000, 715000000, 21700000, 4350000, 2500000, 33200000, 42700000, 112000000, 11800000]
}

# Create DataFrame
df = pd.DataFrame(data)

# Plotting Population Pyramid with a new design
def plot_population_pyramid(df):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Bar plot for male and female population with new colors
    countries = df['Country']
    male_population = df['Male_Population']
    female_population = df['Female_Population']
    
    # Using a new color palette for males and females
    ax.barh(countries, -male_population, color='lightblue', edgecolor='black', hatch='//', label='Male Population')
    ax.barh(countries, female_population, color='lightcoral', edgecolor='black', hatch='xx', label='Female Population')
    
    # Adding grid lines and enhancing plot aesthetics
    ax.grid(True, which='both', axis='x', linestyle='--', color='gray', alpha=0.7)
    
    # Adding labels and formatting
    ax.set_xlabel('Population (in billions)', fontsize=12)
    ax.set_title('Population Pyramid for Conflict-Affected Areas', fontsize=15, fontweight='bold')
    ax.legend()

    # Inverting x-axis for male population
    ax.set_xlim([-max(male_population) * 1.1, max(female_population) * 1.1])

    # Adding custom tick labels for the x-axis
    ax.set_xticks(np.arange(-800000000, 900000000, 200000000))
    ax.set_xticklabels(['800M', '600M', '400M', '200M', '0', '200M', '400M', '600M', '800M'])

    # Add a box around the plot for better readability
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
    
    plt.tight_layout()
    plt.show()

# Call the function to plot the population pyramid with a new design
plot_population_pyramid(df)


# # Cumulative Flow Diagram for Conflict Progression

# In[60]:


import pandas as pd
import matplotlib.pyplot as plt

# Sample DataFrame
# Make sure to replace this with your actual DataFrame
data = {
    'Conflict_Type': ['Type1', 'Type1', 'Type2', 'Type2', 'Type1'],
    'Conflict_Region': ['1', '2', '1', '2', '3']  # Example values that should be numeric
}
df = pd.DataFrame(data)

# Convert 'Conflict_Region' to numeric, forcing errors to NaN
df['Conflict_Region'] = pd.to_numeric(df['Conflict_Region'], errors='coerce')

# Calculate cumulative sum of the Conflict_Region
conflict_progression = df.groupby('Conflict_Type')['Conflict_Region'].cumsum()

# Plotting the cumulative flow diagram
plt.figure(figsize=(10, 6))
plt.plot(df['Conflict_Type'], conflict_progression, label='Cumulative Conflict Progression')
plt.title("Cumulative Flow Diagram of Conflict Progression")
plt.xlabel("Conflict Type")
plt.ylabel("Cumulative Conflict Region")
plt.legend()
plt.show()


# # Coxcomb Chart for UN Resolutions by Country

# In[15]:


import pandas as pd
import plotly.express as px

# Sample DataFrame
data = {
    'Country': ['USA', 'Russia', 'India', 'China', 'Ukraine', 'Israel', 'Palestine', 'France', 'Germany', 'Pakistan', 'Taiwan'],
    'Conflict_Region': ['Mid North America', 'Eastern Europe', 'South Asia', 'East Asia', 'Eastern Europe', 'Middle East', 'Middle East', 'Western Europe', 'Western Europe', 'South West Asia', 'East Asia'],
    'Conflict_Type': ['Tension', 'War', 'Tension', 'Potential Conflict', 'War', 'War', 'War', 'Potential Conflict', 'Tension', 'Extreme Tension', 'Potential Conflict']
}
df = pd.DataFrame(data)

# Count the number of resolutions by Country and Conflict_Type
df_counts = df.groupby(['Country', 'Conflict_Type']).size().reset_index(name='Count')

# Create a Coxcomb (polar line) chart
fig = px.line_polar(df_counts, r='Count', theta='Country', line_close=True)

# Update the figure to fill the area under the line
fig.update_traces(fill='toself')

# Add title and adjust the size (customize the width and height as per your requirement)
fig.update_layout(
    title="Coxcomb Chart of UN Resolutions by Country",
    width=800,  # Set width (e.g., increase or decrease this value as per your needs)
    height=800  # Set height (e.g., increase or decrease this value as per your needs)
)

# Display the figure
fig.show()


# # Timeline Chart for Conflict Events

# In[12]:


import pandas as pd
import plotly.express as px

# Example DataFrame
data = {
    'Country': ['USA', 'Russia', 'India', 'China', 'Ukraine', 'Israel', 'Palestine', 'France', 'Germany', 'Pakistan', 'Taiwan'],
    'Start_Date': ['2020-01-01', '2021-05-01', '2022-07-01', '2022-08-01', '2022-09-01', '2022-10-01', '2022-11-01', '2022-12-01', '2023-01-01', '2023-02-01', '2023-03-01'],
    'End_Date': ['2020-12-31', '2021-12-31', '2022-12-31', '2022-12-31', '2022-12-31', '2022-12-31', '2022-12-31', '2022-12-31', '2023-12-31', '2023-12-31', '2023-12-31'],
    'Conflict_Type': ['Tension', 'War', 'Tension', 'Potential Conflict', 'War', 'War', 'War', 'Potential Conflict', 'Tension', 'Extreme Tension', 'Potential Conflict'],
    'Conflict_Region': ['Mid North America', 'Eastern Europe', 'South Asia', 'East Asia', 'Eastern Europe', 'Middle East', 'Middle East', 'Western Europe', 'Western Europe', 'South West Asia', 'East Asia']
}

df = pd.DataFrame(data)

# Convert Start_Date and End_Date to datetime
df['Start_Date'] = pd.to_datetime(df['Start_Date'])
df['End_Date'] = pd.to_datetime(df['End_Date'])

# Timeline chart to show key events in major conflicts
fig = px.timeline(df, x_start="Start_Date", x_end="End_Date", y="Country",
                  color="Conflict_Type", hover_name="Country")

# Adjust size with width and height as per your requirement
fig.update_layout(
    title="Timeline of Major Conflict Events",
    width=1000,  # Adjust width here (e.g., increase or decrease this value)
    height=600   # Adjust height here (e.g., increase or decrease this value)
)

# Display the figure
fig.show()


# # Chord Diagram for International Alliances

# In[82]:


import numpy as np
import matplotlib.pyplot as plt

# Data for the chord diagram
labels = ["USA", "Russia", "China", "EU", "NATO"]
n = len(labels)

# Create the connections (source, target, values)
source = [0, 1, 2]  # Indices of source nodes
target = [2, 3, 4]  # Indices of target nodes
values = [10, 20, 15]  # Values corresponding to the links

# Prepare the circle layout for the nodes
theta = np.linspace(0, 2 * np.pi, n, endpoint=False)

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

# Plot the nodes
for i in range(n):
    ax.text(theta[i], 0.5, labels[i], horizontalalignment='center', verticalalignment='center', fontsize=12)

# Draw the connections
for s, t, v in zip(source, target, values):
    # Draw lines (chords) between nodes
    ax.plot([theta[s], theta[t]], [0.5, 0.5], linewidth=v, color='b', alpha=0.6)

# Set limits and title
ax.set_ylim(0, 1)
ax.set_title("Chord Diagram of International Alliances", fontsize=16, pad=20)

plt.show()


# # Matrix Plot for Diplomatic Relations

# In[83]:


# Matrix plot for showing diplomatic relations between countries
matrix_data = np.random.rand(10, 10)
plt.matshow(matrix_data, cmap='viridis')
plt.colorbar()
plt.title("Matrix Plot for Diplomatic Relations")
plt.show()


# # Density Plot for Conflict Casualties

# In[86]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample DataFrame for demonstration
data = {
    'Country': ['USA', 'Russia', 'India', 'China', 'Ukraine'],
    'Casualties': [100, 200, 150, 300, 250]
}

df = pd.DataFrame(data)

# Check the columns of the DataFrame
print("Columns in DataFrame:", df.columns)

# If 'Total_Population' doesn't exist, use 'Casualties' or create one
if 'Total_Population' in df.columns:
    population_data = df['Total_Population']
else:
    # Use existing 'Casualties' data for density plot
    population_data = df['Casualties']

# Density plot for visualizing distribution of casualties
plt.figure(figsize=(10, 6))
sns.kdeplot(population_data, shade=True)
plt.title("Density Plot of Conflict Casualties")
plt.xlabel("Casualties")
plt.ylabel("Density")
plt.show()


# # Slope Chart for Conflict Severity over Time

# In[89]:


import pandas as pd
import plotly.express as px

# Sample DataFrame for demonstration
data = {
    'date': ['2020-01', '2020-02', '2020-03', '2020-01', '2020-02', '2020-03'],
    'conflict_severity': [2, 3, 4, 1, 2, 3],
    'conflict_name': ['Conflict A', 'Conflict A', 'Conflict A', 'Conflict B', 'Conflict B', 'Conflict B']
}

df = pd.DataFrame(data)

# Check the columns of the DataFrame
print("Columns in DataFrame:", df.columns)

# Ensure the required columns exist
required_columns = ['date', 'conflict_severity', 'conflict_name']
for col in required_columns:
    if col not in df.columns:
        print(f"Column '{col}' is missing from the DataFrame.")
        # Handle the missing column as needed, e.g., create it or raise an error

# Slope chart to show changes in conflict severity over time
fig = px.line(df, x="date", y="conflict_severity", color="conflict_name", line_shape="spline")

fig.update_layout(title="Slope Chart of Conflict Severity over Time")
fig.show()


# # Treemap for Geopolitical Influence

# In[17]:


import pandas as pd
import plotly.express as px
import numpy as np

# Full dataset
data = {
    'Country': ['USA', 'Russia', 'India', 'China', 'Ukraine', 'Israel', 'Palestine', 'France', 'Germany', 'Pakistan', 'Taiwan'],
    'Conflict_Region': ['Mid North America', 'Eastern Europe', 'South Asia', 'East Asia', 'Eastern Europe', 'Middle East', 'Middle East', 'Western Europe', 'Western Europe', 'South West Asia', 'East Asia'],
    'Conflict_Type': ['Tension', 'War', 'Tension', 'Potential Conflict', 'War', 'War', 'War', 'Potential Conflict', 'Tension', 'Extreme Tension', 'Potential Conflict'],
    'Latitude': [37.0902, 61.5240, 20.5937, 35.8617, 48.3794, 31.0461, 31.9522, 46.6034, 51.1657, 30.3753, 23.6978],
    'Longitude': [-95.7129, 105.3188, 78.9629, 104.1954, 31.1656, 34.8516, 35.2332, 1.8883, 10.4515, 69.3451, 121.0200],
    'Altitude': [760, 600, 160, 1840, 175, 508, 795, 375, 263, 900, 1150],
    'Conflict_Intensity': [10, 20, 30, 25, 15, 18, 5, 12, 22, 16, 14],
    'Deaths': np.random.randint(1000, 50000, size=11),
    'Economic_Impact_Billion': np.random.uniform(1.5, 100, size=11),
    'Environmental_Damage_Index': np.random.uniform(1, 10, size=11),
    'UN_Interventions': np.random.choice([1, 2, 3, 4], size=11),
    'Total_Population': [331002651, 145912025, 1380004385, 1439323776, 43733762, 8655535, 5000000, 65273511, 83783942, 225199937, 23816775],
    'Male_Population': [162000000, 67000000, 705000000, 724000000, 22000000, 4300000, 2500000, 32000000, 41000000, 113000000, 12000000],
    'Female_Population': [169000000, 78900000, 675000000, 715000000, 21700000, 4350000, 2500000, 33200000, 42700000, 112000000, 11800000]
}

# Create DataFrame
df = pd.DataFrame(data)

# Treemap showing geopolitical influence by country and region
fig = px.treemap(df, path=['Country', 'Conflict_Region'], values='Conflict_Intensity',
                 color='Conflict_Intensity', hover_name='Country')

# Adjust size with width and height as per your requirement
fig.update_layout(
    title="Treemap of Geopolitical Influence by Region and Country",
    width=1200,  # Adjust width here (e.g., increase or decrease this value)
    height=800   # Adjust height here (e.g., increase or decrease this value)
)

# Display the figure
fig.show()


# # NLP-Based Sentiment Analysis

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Sample data: list of texts (UN speeches/reports)
reports = [
    "The UN stands firm in its commitment to peace.",
    "Conflict continues to plague the region, causing suffering.",
    "We celebrate the progress made towards sustainable development.",
    "There is an urgent need for humanitarian assistance.",
    "International cooperation is essential for success."
]

# Initialize Sentiment Analyzer
sid = SentimentIntensityAnalyzer()

# Perform sentiment analysis
sentiments = [sid.polarity_scores(report) for report in reports]

# Convert to DataFrame for visualization
sentiment_df = pd.DataFrame(sentiments)

# Visualize sentiment analysis
sentiment_df[['neg', 'neu', 'pos']].plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title("Sentiment Analysis of UN Reports")
plt.xlabel("Reports")
plt.ylabel("Sentiment Score")
plt.xticks(ticks=range(len(reports)), labels=[f'Report {i+1}' for i in range(len(reports))], rotation=0)
plt.show()


# # Pareto Chart

# In[18]:


import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

# Actual conflict-related factors
factors = ['Geopolitical Influence', 'Conflict Intensity', 'Economic Impact', 'Environmental Damage']

# Sample contributions data (can be adjusted based on the dataset)
contributions = [60, 25, 10, 5]

# Pareto plot
def plot_pareto_chart(width=8, height=5):
    fig, ax = plt.subplots(figsize=(width, height))  # Set figure size dynamically
    ax.bar(factors, contributions, color="C0", label='Contributions')
    
    # Calculate cumulative percentages for Pareto line
    cumulative_percent = np.cumsum(contributions) / sum(contributions) * 100
    ax2 = ax.twinx()  # Create a twin Axes sharing the x-axis
    ax2.plot(factors, cumulative_percent, color="C1", marker="D", ms=7, label='Cumulative Percentage')
    
    # Format y-axis of ax2 as percentage
    ax2.yaxis.set_major_formatter(PercentFormatter())
    
    # Titles and labels
    plt.title("Pareto Chart: Global Conflict Factors")
    ax.set_xlabel("Factors")
    ax.set_ylabel("Contributions")
    
    # Legends
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    # Show plot
    plt.show()

# Example of calling the function with dynamic size
plot_pareto_chart(width=10, height=6)  # Adjust the width and height as per your requirement


# # Error Bar

# In[7]:


import matplotlib.pyplot as plt
import numpy as np

# Sample data
data = {
    'x': np.array([1, 2, 3, 4, 5]),            # x-values
    'y': np.array([2.5, 3.0, 4.0, 5.2, 6.5]),   # y-values
    'error': np.array([0.5, 0.2, 0.3, 0.4, 0.6])  # error values for y
}

# Create an error bar plot
plt.errorbar(x=data['x'], y=data['y'], yerr=data['error'], fmt='o', color='b', capsize=5)
plt.title("Error Bar Plot")
plt.xlabel("X-axis Label")
plt.ylabel("Y-axis Label")
plt.grid(True)  # Add a grid for better readability
plt.show()


# # Density Plot

# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Sample data
np.random.seed(42)  # For reproducibility
data = {
    'feature': np.random.normal(loc=0, scale=1, size=1000)  # Generate 1000 random values from a normal distribution
}

# Create a density plot
sns.kdeplot(data['feature'], shade=True, color='blue')
plt.title("Density Plot")
plt.xlabel("Feature Value")
plt.ylabel("Density")
plt.grid(True)  # Add a grid for better readability
plt.show()


# # Funnel Chart

# In[1]:


import plotly.graph_objects as go

# Actual stages in the conflict resolution process
stages = ["Conflict Identification", "UN Diplomacy Intervention", "Peace Negotiations", "Post-Conflict Recovery"]

# Corresponding values for each stage (example data)
stage_values = [500, 450, 300, 200]

# Create a funnel chart
fig = go.Figure(go.Funnel(
    y=stages,
    x=stage_values
))

# Show the figure
fig.update_layout(title="Funnel Chart: Conflict Resolution Stages")
fig.show()


# #  Spiral Chart

# In[2]:


import numpy as np
import matplotlib.pyplot as plt  # Importing matplotlib.pyplot as plt

theta = np.linspace(0, 4*np.pi, 100)
r = np.linspace(0, 1, 100)

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta, r)
plt.title("Spiral Chart")
plt.show()


# #  NLP-based Sentiment Analysis of Social Media during Conflicts

# In[2]:


pip install spacy pydantic vaderSentiment


# In[4]:


# Import necessary libraries
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt

# Initialize the sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Sample DataFrame with social media posts (Replace with actual dataset)
data = {'post': ["This war is tragic", "Peace talks are hopeful", "Conflict is escalating"]}
df = pd.DataFrame(data)

# Function for sentiment analysis
def analyze_sentiment(text):
    sentiment = analyzer.polarity_scores(text)
    return sentiment['compound']

# Apply the sentiment analysis function to each post
df['sentiment'] = df['post'].apply(analyze_sentiment)

# Plot sentiment (assuming you have date information in the real dataset)
plt.plot(df['sentiment'])
plt.title('Sentiment Analysis of Social Media Posts')
plt.xlabel('Post Number')
plt.ylabel('Sentiment Score')
plt.show()


# # Correlation Analysis of Geographical Features and Conflict Occurrence

# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Data initialization
data = {
    'Country': ['USA', 'Russia', 'India', 'China', 'Ukraine', 'Israel', 'Palestine', 'France', 'Germany', 'Pakistan', 'Taiwan'],
    'Conflict_Region': ['Mid North America', 'Eastern Europe', 'South Asia', 'East Asia', 'Eastern Europe', 'Middle East', 'Middle East', 'Western Europe', 'Western Europe', 'South West Asia', 'East Asia'],
    'Conflict_Type': ['Tension', 'War', 'Tension', 'Potential Conflict', 'War', 'War', 'War', 'Potential Conflict', 'Tension', 'Extreme Tension', 'Potential Conflict'],
    'Latitude': [37.0902, 61.5240, 20.5937, 35.8617, 48.3794, 31.0461, 31.9522, 46.6034, 51.1657, 30.3753, 23.6978],
    'Longitude': [-95.7129, 105.3188, 78.9629, 104.1954, 31.1656, 34.8516, 35.2332, 1.8883, 10.4515, 69.3451, 121.0200],
    'Altitude': [760, 600, 160, 1840, 175, 508, 795, 375, 263, 900, 1150],
    'Conflict_Intensity': [10, 20, 30, 25, 15, 18, 5, 12, 22, 16, 14],
    'Deaths': np.random.randint(1000, 50000, size=11),  
    'Economic_Impact_Billion': np.random.uniform(1.5, 100, size=11),
    'Environmental_Damage_Index': np.random.uniform(1, 10, size=11),
    'UN_Interventions': np.random.choice([1, 2, 3, 4], size=11),
    'Total_Population': [331002651, 145912025, 1380004385, 1439323776, 43733762, 8655535, 5000000, 65273511, 83783942, 225199937, 23816775],
    'Male_Population': [162000000, 67000000, 705000000, 724000000, 22000000, 4300000, 2500000, 32000000, 41000000, 113000000, 12000000],
    'Female_Population': [169000000, 78900000, 675000000, 715000000, 21700000, 4350000, 2500000, 33200000, 42700000, 112000000, 11800000]
}

# Creating DataFrame
df = pd.DataFrame(data)

# Calculate the correlation matrix for geographical features and conflict-related variables
correlation_matrix = df[['Latitude', 'Longitude', 'Altitude', 'Conflict_Intensity', 'Deaths', 'Economic_Impact_Billion', 'Environmental_Damage_Index']].corr()

# Display the correlation matrix
print("\nCorrelation Matrix:\n", correlation_matrix)

# Visualization: Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap of Geographical Features and Conflict Occurrence")
plt.show()

# Pairplot for advanced data visualization
sns.pairplot(df[['Latitude', 'Longitude', 'Altitude', 'Conflict_Intensity', 'Deaths', 'Economic_Impact_Billion', 'Environmental_Damage_Index']])
plt.suptitle("Pairplot for Geographical Features and Conflict Analysis", y=1.02)
plt.show()


# In[ ]:





# In[ ]:





# # Interactive Dashboards for Conflict Analysis

# In[11]:


import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np

# Step 1: Prepare the data
data = {
    'Country': ['USA', 'Russia', 'India', 'China', 'Ukraine', 'Israel', 'Palestine', 'France', 'Germany', 'Pakistan', 'Taiwan'],
    'Conflict_Region': ['Mid North America', 'Eastern Europe', 'South Asia', 'East Asia', 'Eastern Europe', 'Middle East', 'Middle East', 'Western Europe', 'Western Europe', 'South West Asia', 'East Asia'],
    'Conflict_Type': ['Tension', 'War', 'Tension', 'Potential Conflict', 'War', 'War', 'War', 'Potential Conflict', 'Tension', 'Extreme Tension', 'Potential Conflict'],
    'Latitude': [37.0902, 61.5240, 20.5937, 35.8617, 48.3794, 31.0461, 31.9522, 46.6034, 51.1657, 30.3753, 23.6978],
    'Longitude': [-95.7129, 105.3188, 78.9629, 104.1954, 31.1656, 34.8516, 35.2332, 1.8883, 10.4515, 69.3451, 121.0200],
    'Altitude': [760, 600, 160, 1840, 175, 508, 795, 375, 263, 900, 1150],
    'Conflict_Intensity': [10, 20, 30, 25, 15, 18, 5, 12, 22, 16, 14],
    'Deaths': np.random.randint(1000, 50000, size=11),
    'Economic_Impact_Billion': np.random.uniform(1.5, 100, size=11),
    'Environmental_Damage_Index': np.random.uniform(1, 10, size=11),
    'UN_Interventions': np.random.choice([1, 2, 3, 4], size=11),
    'Total_Population': [331002651, 145912025, 1380004385, 1439323776, 43733762, 8655535, 5000000, 65273511, 83783942, 225199937, 23816775],
    'Male_Population': [162000000, 67000000, 705000000, 724000000, 22000000, 4300000, 2500000, 32000000, 41000000, 113000000, 12000000],
    'Female_Population': [169000000, 78900000, 675000000, 715000000, 21700000, 4350000, 2500000, 33200000, 42700000, 112000000, 11800000]
}

df = pd.DataFrame(data)

# Step 2: Create Choropleth map using Plotly
fig = px.choropleth(
    df,
    locations="Country",
    locationmode="country names",
    color="Conflict_Intensity",
    hover_name="Country",
    hover_data=["Conflict_Region", "Conflict_Type", "Deaths", "Economic_Impact_Billion", "Environmental_Damage_Index", "UN_Interventions"],
    color_continuous_scale=px.colors.sequential.Plasma,
    title="Global Conflict Intensity and Analysis"
)

# Step 3: Set up the Dash application
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Interactive Global Conflict Mapping Dashboard", style={'text-align': 'center'}),
    
    # Choropleth Map
    dcc.Graph(
        id='conflict_map',
        figure=fig
    ),
    
    # Dropdown for user to filter by conflict type
    dcc.Dropdown(
        id='conflict_type_dropdown',
        options=[{'label': conflict, 'value': conflict} for conflict in df['Conflict_Type'].unique()],
        value='War',
        placeholder="Select a Conflict Type",
        style={"width": "50%", 'display': 'inline-block'}
    ),
    
    # Output description text for the selected conflict
    html.Div(id='output_container', children=[])
])

# Step 4: Callback to update the choropleth map based on the selected conflict type
@app.callback(
    [Output(component_id='conflict_map', component_property='figure'),
     Output(component_id='output_container', component_property='children')],
    [Input(component_id='conflict_type_dropdown', component_property='value')]
)
def update_graph(selected_conflict):
    filtered_df = df[df['Conflict_Type'] == selected_conflict]

    # Update the map with the filtered data
    fig = px.choropleth(
        filtered_df,
        locations="Country",
        locationmode="country names",
        color="Conflict_Intensity",
        hover_name="Country",
        hover_data=["Conflict_Region", "Conflict_Type", "Deaths", "Economic_Impact_Billion", "Environmental_Damage_Index", "UN_Interventions"],
        color_continuous_scale=px.colors.sequential.Plasma,
        title=f"Global Conflict Intensity: {selected_conflict}"
    )

    return fig, f"Showing conflict analysis for: {selected_conflict}"

# Step 5: Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)


# # Conflict Prediction using LSTM Models

# In[13]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Sample DataFrame (Replace with actual data)
data = {'conflict_intensity': [5, 8, 10, 7, 6, 9, 8]}
df = pd.DataFrame(data)

# Prepare the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df['conflict_intensity'].values.reshape(-1, 1))

# Create sequences for LSTM
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

seq_length = 3
X = create_sequences(scaled_data, seq_length)
y = scaled_data[seq_length:]

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=10, batch_size=1)

# Predictions (replace with actual test data)
predicted_conflicts = model.predict(X)
predicted_conflicts = scaler.inverse_transform(predicted_conflicts)


# # 3D Visualization of Military Alliances and Coalitions

# In[10]:


import pandas as pd
import plotly.graph_objects as go

# Sample DataFrame (Replace with actual data)
data = {
    'country': ['USA', 'Russia', 'India', 'Brazil', 'China', 'Australia'],
    'alliance': ['NATO', 'BRICS', 'BRICS', 'BRICS', 'BRICS', 'AUKUS']
}
df = pd.DataFrame(data)

# Create a 3D choropleth map
fig = go.Figure(go.Choroplethmapbox(
    geojson='https://raw.githubusercontent.com/datasets/geo-boundaries/master/data/countries.geojson',
    locations=df['country'],
    z=df['alliance'].factorize()[0],
    colorscale='Viridis',
    text=df['country'],  # Show country names on the map
    hoverinfo='text',  # Display country names on hover
))

# Update layout for mapbox
fig.update_layout(
    mapbox_style="open-street-map",  # Change to Open Street Map for a street map feel
    mapbox_zoom=1.5,  # Adjust zoom level
    mapbox_center={"lat": 20, "lon": 0},  # Center of the map
    margin={"r":0,"t":0,"l":0,"b":0},  # Remove margins
)

# Change ocean color to blue
fig.update_traces(marker=dict(line=dict(width=0)),  # No borders between countries
                  selector=dict(type='choroplethmapbox'))

# Add a color scale for ocean
fig.add_trace(go.Choroplethmapbox(
    geojson='https://raw.githubusercontent.com/datasets/geo-boundaries/master/data/countries.geojson',
    locations=df['country'],
    z=[1] * len(df),  # Dummy variable to set the color of countries
    colorscale=[[0, 'blue'], [1, 'blue']],  # Set ocean color to blue
    showscale=False  # Hide scale
))

# Adjust the layout to fit the whole output screen
fig.update_layout(
    height=700,  # Adjust height as needed
    width=1500,  # Adjust width as needed
)

# Show the figure
fig.show()


# In[ ]:





# # Comparison of UN Resolutions Impact

# In[19]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load UN resolution impact data
df_resolutions = pd.read_csv(r'E:\un_resolutions.csv')

# Add 'unsc_resolution' column (Example: 'Yes' or 'No')
df_resolutions['unsc_resolution'] = ['Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 
                                     'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No']

# Calculate reduction in conflict intensity after resolutions
df_resolutions['conflict_intensity_reduction'] = df_resolutions['before_resolution_intensity'] - df_resolutions['after_resolution_intensity']

# Visualize the comparison
plt.figure(figsize=(10, 6))
sns.barplot(data=df_resolutions, x='conflict', y='conflict_intensity_reduction', hue='unsc_resolution')
plt.title('Impact of UN Resolutions on Conflict Intensity Reduction')
plt.xlabel('Conflict')
plt.ylabel('Reduction in Intensity')
plt.xticks(rotation=45)
plt.show()


# # Conflict Timeline Visualization

# In[9]:


import os
import pandas as pd
import plotly.express as px

# Define the file path
csv_file_path = 'conflict_timeline.csv'

# Step 1: Create the CSV file with sample data if it doesn't exist
if not os.path.exists(csv_file_path):
    # Sample data for the conflict timeline
    data = {
        'country': ['USA', 'Russia', 'India', 'China', 'Ukraine', 'Israel', 'Palestine', 'France', 'Germany', 'Pakistan', 'Taiwan'],
        'conflict': ['Tension', 'War', 'Tension', 'Potential Conflict', 'War', 'War', 'War', 'Potential Conflict', 'Tension', 'Extreme Tension', 'Potential Conflict'],
        'intensity': [10, 20, 30, 25, 15, 18, 5, 12, 22, 16, 14],
        'year': [2000, 2005, 2010, 2015, 2022, 2023, 2021, 2015, 2022, 2019, 2021],
    }
    
    # Create a DataFrame
    df_sample = pd.DataFrame(data)
    
    # Save to CSV
    df_sample.to_csv(csv_file_path, index=False)
    print(f"Created {csv_file_path} with sample data.")
else:
    print(f"{csv_file_path} already exists.")

# Step 2: Load the conflict timeline data
df_timeline = pd.read_csv(csv_file_path)

# Step 3: Create an animated scatter plot to show conflict progression
fig = px.scatter_geo(df_timeline, locations="country", locationmode="country names", 
                     color="conflict", hover_name="conflict", size="intensity",
                     animation_frame="year", projection="natural earth")

# Customize map colors and settings
fig.update_geos(
    showcoastlines=True, coastlinecolor="black",
    showland=True, landcolor="lightgreen",   # Light green for continents
    showocean=True, oceancolor="lightblue",  # Light blue for oceans
    showlakes=True, lakecolor="blue",
    showrivers=True, rivercolor="blue",
    projection_scale=1,                      # Controls zoom level
    showcountries=True, countrycolor="black",
    visible=True
)

# Ensure country names are visible upon zoom
fig.update_layout(
    title="Timeline of Global Conflicts",
    width=1200,  # Set your desired width (in pixels)
    height=800,  # Set your desired height (in pixels)
    geo=dict(
        showframe=False,
        showcoastlines=True,
        projection_type='natural earth',
        showocean=True,
        oceancolor="lightblue",
        showland=True,
        landcolor="lightgreen",
        countrycolor="black",
        showcountries=True
    )
)

fig.show()


# # Natural Resource Competition as a Conflict Factor

# In[29]:


import os
import pandas as pd
import statsmodels.api as sm

# Define the file path
csv_file_path = 'resource_conflict_data.csv'

# Step 1: Create the CSV file with sample data if it doesn't exist
if not os.path.exists(csv_file_path):
    # Sample data for resource conflict analysis
    data = {
        'water_access': [90, 85, 80, 70, 60, 50],   # Percentage of population with access to clean water
        'oil_production': [1000, 1500, 2000, 3000, 4000, 5000],  # Barrels of oil produced per day
        'conflict_intensity': [2, 3, 4, 5, 6, 7]  # Conflict intensity score (1-10)
    }
    
    # Create a DataFrame
    df_sample = pd.DataFrame(data)
    
    # Save to CSV
    df_sample.to_csv(csv_file_path, index=False)
    print(f"Created {csv_file_path} with sample data.")
else:
    print(f"{csv_file_path} already exists.")

# Step 2: Load the dataset
df_resource_conflict = pd.read_csv(csv_file_path)

# Step 3: Prepare independent and dependent variables for regression
X = df_resource_conflict[['water_access', 'oil_production']]  # Independent variables
y = df_resource_conflict['conflict_intensity']  # Dependent variable

# Step 4: Add a constant to the model
X = sm.add_constant(X)

# Step 5: Fit the regression model
model = sm.OLS(y, X).fit()

# Step 6: Print the summary of regression results
print(model.summary())


# # Resource Allocation Optimization using ML

# In[31]:


import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Define the file path
csv_file_path = 'aid_allocation.csv'

# Step 1: Create the CSV file with sample data if it doesn't exist
if not os.path.exists(csv_file_path):
    # Sample data for aid allocation analysis
    data = {
        'population': [100000, 200000, 150000, 300000, 250000, 400000],
        'conflict_intensity': [3, 5, 4, 6, 7, 8],  # Conflict intensity score (1-10)
        'infrastructure_damage': [100, 200, 150, 300, 250, 400],  # Damage cost in thousands
        'aid_allocation': [5000, 10000, 7000, 12000, 9000, 15000]  # Aid allocation in dollars
    }
    
    # Create a DataFrame
    df_sample = pd.DataFrame(data)
    
    # Save to CSV
    df_sample.to_csv(csv_file_path, index=False)
    print(f"Created {csv_file_path} with sample data:\n{df_sample}")
else:
    print(f"{csv_file_path} already exists.")

# Step 2: Load the dataset
df_aid = pd.read_csv(csv_file_path)

# Step 3: Define features and target variable
X = df_aid[['population', 'conflict_intensity', 'infrastructure_damage']]  # Features
y = df_aid['aid_allocation']  # Target variable

# Step 4: Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
xgb_model.fit(X_train, y_train)

# Step 6: Predict and evaluate
y_pred = xgb_model.predict(X_test)
print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')


# # Humanitarian Aid Impact Prediction:

# In[35]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Step 1: Create a sample dataset
data = {
    'aid_amount': [10000, 20000, 15000, 30000, 25000],
    'conflict_duration': [12, 8, 15, 24, 20],  # in months
    'region': ['Africa', 'Asia', 'Africa', 'Asia', 'Europe'],
    'UN_involvement': [1, 0, 1, 0, 1]  # 1 for involved, 0 for not involved
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
csv_file_path = 'humanitarian_aid_data.csv'
df.to_csv(csv_file_path, index=False)

# Display the DataFrame
print("Sample data saved to humanitarian_aid_data.csv:")
print(df)

# Step 2: Feature and target selection using the DataFrame
X = df[['aid_amount', 'region', 'UN_involvement']]  # Use df instead of data
y = df['conflict_duration']  # Target variable (conflict duration)

# Convert categorical variable 'region' to dummy variables
X = pd.get_dummies(X, columns=['region'], drop_first=True)

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 5: Predict on the test set
y_pred = rf_model.predict(X_test)

# Step 6: Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")


# #  Global Migration Patterns Post-Conflict

# In[12]:


import geopandas as gpd
import pandas as pd
import os
import matplotlib.pyplot as plt

# Step 1: Create the 'migration_data.csv' file and save it to "E:\"
# Sample migration data with columns: origin, destination, longitude, latitude, and migration_count
data = {
    'origin': ['USA', 'Russia', 'India', 'China', 'Germany'] * 4,
    'destination': ['North Korea', 'Ukraine', 'Israel', 'Taiwan', 'France'] * 4,
    'longitude': [-99.1332, 0.1276, 31.2357, 2.3522, 139.6917] * 4,  # Approximate longitudes of cities
    'latitude': [19.4326, 51.5074, 30.0444, 48.8566, 35.6895] * 4,   # Approximate latitudes of cities
    'migration_count': [1500, 2200, 1800, 2500, 3000] * 4
}

# Convert the dictionary to a DataFrame
migration_data = pd.DataFrame(data)

# Save the migration data to CSV format in "E:\" location
csv_path = "E:/migration_data.csv"
migration_data.to_csv(csv_path, index=False)

# Verify if the file has been saved
if os.path.exists(csv_path):
    print(f"'migration_data.csv' has been successfully saved at {csv_path}")
else:
    print(f"Failed to save 'migration_data.csv' at {csv_path}")

# Display the DataFrame
print("\nMigration Data DataFrame:\n", migration_data)

# Step 2: Download the 'naturalearth_lowres' dataset for offline use
# Load the naturalearth_lowres dataset using Geopandas
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Save the dataset to "E:\" location in shapefile format
shp_save_path = "E:/naturalearth_lowres.shp"
world.to_file(shp_save_path)

# Verify if the file has been saved
if os.path.exists(shp_save_path):
    print(f"'naturalearth_lowres' dataset has been successfully saved at {shp_save_path}")
else:
    print(f"Failed to save 'naturalearth_lowres' dataset at {shp_save_path}")

# Plot the world map using the dataset and display it
ax = world.plot(figsize=(10, 6))
ax.set_title("World Map (Natural Earth Low Resolution)")
plt.show()


# In[11]:


import geopandas as gpd
import os

# Load the naturalearth_lowres dataset
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Define the file path to save the dataset in 'E:\'
save_path = "E:/naturalearth_lowres.shp"

# Save the dataset as a shapefile
world.to_file(save_path)

# Verify if the file has been saved
if os.path.exists(save_path):
    print(f"The dataset has been successfully saved at {save_path}")
else:
    print(f"Failed to save the dataset at {save_path}")

# Display the world map using the dataset
ax = world.plot(figsize=(10, 6))
ax.set_title("World Map (Natural Earth Low Resolution)")
plt.show()


# # Custom Machine Learning Model for Conflict Prediction

# In[44]:


import pandas as pd
import xgboost as xgb
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv(r'E:\conflict_data.csv')

# Display the complete data
print("Complete Data from CSV:")
print(data)

# Encode categorical columns
label_encoder = LabelEncoder()
data['Conflict_Region'] = label_encoder.fit_transform(data['Conflict_Region'])  # Encode the target column
for col in data.select_dtypes(include=['object']).columns:  # Encode all other object columns
    data[col] = label_encoder.fit_transform(data[col])

# Define features and target
X = data.drop('Conflict_Region', axis=1)
y = data['Conflict_Region']

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
rf_model = RandomForestClassifier(n_estimators=100)
xgb_model = xgb.XGBClassifier()

# LSTM model expects 3D input: [samples, timesteps, features]
# Reshape X_train and X_test for LSTM
X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

lstm_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1], 1)),  # Input layer using Input() object
    tf.keras.layers.LSTM(50, return_sequences=False),  # LSTM layer
    tf.keras.layers.Dense(1)  # Output layer
])

# Compile LSTM model
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, verbose=0)

# Ensemble model (LSTM excluded here since VotingClassifier only works with traditional ML models)
ensemble_model = VotingClassifier(estimators=[('rf', rf_model), ('xgb', xgb_model)], voting='hard')
ensemble_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = ensemble_model.predict(X_test)
print(f'Ensemble Model Accuracy: {accuracy_score(y_test, y_pred)}')

# Predict and evaluate LSTM model
y_pred_lstm = lstm_model.predict(X_test_lstm)
y_pred_lstm_classes = (y_pred_lstm > 0.5).astype(int)  # Binarizing the output for binary classification
print(f'LSTM Model Accuracy: {accuracy_score(y_test, y_pred_lstm_classes)}')


# #  Visualizing Trade Disruptions in Conflict Zones

# In[17]:


# Trade Disruptions in Conflict Zones using Plotly Sankey Diagram

import plotly.graph_objects as go

# Define Sankey diagram nodes and links for trade before and after conflict
labels = ["USA", "Russia", "India", "China", "Middle East"]
sources = [0, 1, 2, 2, 3]  # Pre-conflict trade sources
targets = [3, 3, 3, 4, 4]  # Post-conflict trade disruptions
values = [100, 50, 75, 20, 30]  # Trade volumes

# Create the Sankey diagram
fig = go.Figure(go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=labels
    ),
    link=dict(
        source=sources,
        target=targets,
        value=values
    )
))

fig.update_layout(title_text="Trade Disruptions Due to Conflict", font_size=10)
fig.show()


# # Data Privacy Considerations in Conflict Analysis

# In[40]:


import pandas as pd
import plotly.express as px

# Sample data for Global Conflict Mapping
data = {
    'Country': [
        'United States', 'Russia', 'China', 'India', 'Germany', 'France',
        'United Kingdom', 'Brazil', 'Japan', 'South Africa', 'Canada', 
        'Australia', 'Mexico', 'Italy', 'South Korea', 'Argentina', 
        'Turkey', 'Saudi Arabia', 'Spain', 'Netherlands'
    ],
    'Capital': [
        'Washington D.C.', 'Moscow', 'Beijing', 'New Delhi', 'Berlin', 
        'Paris', 'London', 'Brasília', 'Tokyo', 'Pretoria', 'Ottawa', 
        'Canberra', 'Mexico City', 'Rome', 'Seoul', 'Buenos Aires', 
        'Ankara', 'Riyadh', 'Madrid', 'Amsterdam'
    ],
    'Conflict_Severity': [5, 7, 4, 6, 3, 4, 5, 2, 6, 4, 3, 5, 6, 2, 5, 3, 7, 4, 3, 2],
    'Latitude': [
        37.0902, 61.5240, 35.8617, 20.5937, 51.1657, 46.6034,
        55.3781, -14.2350, 36.2048, -30.5595, 56.1304, -25.2744,
        23.6345, 41.8719, 35.9078, -38.4161, 38.9968, 40.4637,
        39.9334, 52.1326
    ],
    'Longitude': [
        -95.7129, 105.3188, 104.1954, 78.9629, 10.4515, 1.8883,
        -3.4360, -51.9253, 138.2529, 22.9375, -106.3468, 133.7751,
        -102.5528, 12.5674, 127.7669, 127.7669, 45.0943, 39.0742,
        -3.7038, 5.2913
    ],
    'Data_Confidentiality': [
        'High', 'Medium', 'Medium', 'High', 'Low', 'Medium',
        'High', 'Low', 'High', 'Medium', 'High', 'Medium',
        'Medium', 'Low', 'Medium', 'Low', 'High', 'High',
        'Low', 'Medium'
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Display the DataFrame in the output
print("Global Conflict Data:")
print(df)

# Create a political map using Plotly
fig = px.choropleth(
    df,
    locations='Country',  # We use country names for locations
    locationmode='country names',  # Use country names
    color='Conflict_Severity',  # Color based on conflict severity
    hover_name='Country',  # Display country name on hover
    hover_data=['Capital', 'Data_Confidentiality'],  # Additional hover info
    title='Global Political Map with Conflict Severity',
    color_continuous_scale=px.colors.sequential.Plasma  # Color scale
)

# Update layout to center the map and allow resizing
fig.update_geos(
    visible=False,  # Hide base geo features
    showcountries=True,  # Show country boundaries
    countrycolor="Black"  # Color for country borders
)

# Center the title and adjust size
fig.update_layout(
    autosize=True,
    width=1500,
    height=800,
    title_x=0.5,  # Center the title
    margin=dict(l=50, r=50, t=50, b=50),  # Set margins for centering
)

# Show the figure
fig.show()


# # Global Conflict Severity Ranking using Machine Learning

# In[25]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load conflict dataset
data = pd.read_csv(r'E:\conflict_data.csv')

# Encode categorical variables
data = pd.get_dummies(data, columns=['Conflict_Type'], drop_first=True)

# Selecting features and target variable
X = data[['UN_Interventions', 'Economic_Impact_Billion'] + list(data.columns[data.columns.str.startswith('Conflict_Type')])]
y = data['Conflict_Intensity']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# Predict and rank conflicts
y_pred = rf.predict(X_test)

# Display predictions
print(y_pred)


# # Conflict Resolution Techniques Visualization

# In[27]:


import plotly.graph_objects as go

# Data: Mediation, sanctions, peacekeeping effectiveness
resolution_techniques = ['Mediation', 'Sanctions', 'Peacekeeping']
success_rates = [70, 55, 80]  # Example success rates

# Radar chart to visualize effectiveness of conflict resolution techniques
fig = go.Figure(data=go.Scatterpolar(
    r=success_rates,
    theta=resolution_techniques,
    fill='toself'
))

fig.update_layout(
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 100])
    ),
    showlegend=False,
    title='Conflict Resolution Techniques Effectiveness',
    width=700,  # Set the desired width
    height=700  # Set the desired height
)

fig.show()


# # Prediction of Military Movements using Geospatial Data

# In[2]:


import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import OneHotEncoder
import plotly.express as px  # Make sure to import Plotly Express

# Create a sample dataset for military movements
data = {
    'latitude': [34.05, 34.56, 35.23, 36.11, 37.44, 38.22, 39.01, 40.35, 41.12, 42.48,
                 43.07, 44.16, 45.09, 46.33, 47.89, 48.12, 49.47, 50.03, 51.11, 52.44],
    'longitude': [-118.25, -117.65, -116.67, -115.98, -114.74, -113.87, -112.65, -111.12, -110.34, -109.45,
                  -108.95, -107.12, -106.53, -105.78, -104.92, -103.56, -102.78, -101.34, -100.67, -99.45],
    'terrain': ['mountain', 'desert', 'forest', 'plain', 'urban', 'hilly', 'coastal', 'flat', 'swamp', 'rural',
                'mountain', 'desert', 'forest', 'plain', 'urban', 'hilly', 'coastal', 'flat', 'swamp', 'rural'],
    'weather': ['sunny', 'cloudy', 'rainy', 'sunny', 'foggy', 'stormy', 'sunny', 'cloudy', 'rainy', 'sunny',
                'foggy', 'stormy', 'sunny', 'cloudy', 'rainy', 'sunny', 'foggy', 'stormy', 'sunny', 'cloudy'],
    'future_movement': [1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 
                        0, 1, 0, 0, 1, 1, 1, 0, 0, 1],
    'unit_type': ['infantry', 'armored', 'airborne', 'infantry', 'naval', 'armored', 'airborne', 'infantry', 'naval', 'armored',
                  'airborne', 'infantry', 'naval', 'armored', 'airborne', 'infantry', 'naval', 'armored', 'airborne', 'infantry'],
    'logistics_support': [5, 3, 4, 6, 2, 5, 7, 4, 3, 6,
                          2, 5, 4, 6, 3, 4, 5, 2, 6, 5],
    'mission_type': ['reconnaissance', 'combat', 'support', 'reconnaissance', 'combat', 'support', 'reconnaissance', 'combat', 'support', 'reconnaissance',
                     'combat', 'support', 'reconnaissance', 'combat', 'support', 'reconnaissance', 'combat', 'support', 'reconnaissance', 'combat'],
    'time_of_day': ['morning', 'afternoon', 'evening', 'morning', 'afternoon', 'evening', 'morning', 'afternoon', 'evening', 'morning',
                    'afternoon', 'evening', 'morning', 'afternoon', 'evening', 'morning', 'afternoon', 'evening', 'morning', 'afternoon'],
}

# Create a DataFrame
military_movements_df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
csv_file_path = r'E:\military_movements.csv'
military_movements_df.to_csv(csv_file_path, index=False)

# Load the data for LSTM
data = pd.read_csv(r'E:\military_movements.csv')

# Preprocess categorical data using One-Hot Encoding
encoder = OneHotEncoder(sparse_output=False)  # Use sparse_output instead of sparse
encoded_features = encoder.fit_transform(data[['terrain', 'weather', 'unit_type', 'mission_type', 'time_of_day']])
encoded_feature_names = encoder.get_feature_names_out(['terrain', 'weather', 'unit_type', 'mission_type', 'time_of_day'])

# Create a new DataFrame with the encoded features
encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)

# Combine the numeric and encoded features
X = pd.concat([data[['latitude', 'longitude', 'logistics_support']], encoded_df], axis=1).values.reshape(-1, 1, encoded_df.shape[1] + 3)
y = data['future_movement'].values

# Building the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(1, X.shape[2])))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(1, activation='linear'))

# Compile and fit the model
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=10, batch_size=32)

# Create a sunburst chart
fig = px.sunburst(data, path=['latitude', 'longitude', 'unit_type', 'terrain', 'weather', 'mission_type', 'logistics_support', 'time_of_day'], values='future_movement', title="Military Movements Sunburst Chart")

# Update the layout to adjust the size of the chart
fig.update_layout(width=800, height=800)  # Change width and height as desired

# Show the figure
fig.show()


# # Topic Modeling of UN Speeches using Gensim

# In[35]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim import corpora
from gensim.models import LdaModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Uncomment the following lines to download NLTK resources if needed
# nltk.download('punkt')
# nltk.download('stopwords')

# Sample dataset with country statements
data = {
    'Country': [
        'United States', 'Russia', 'China', 'India', 'Germany', 'France',
        'United Kingdom', 'Brazil', 'Japan', 'South Africa', 'Canada', 
        'Australia', 'Mexico', 'Italy', 'South Korea', 'Argentina', 
        'Turkey', 'Saudi Arabia', 'Spain', 'Netherlands'
    ],
    'Statement': [
        "We must work together to promote peace and security.",
        "The situation in the region requires immediate attention.",
        "Collaboration is key to address global challenges.",
        "Our commitment to climate action must be reinforced.",
        "Human rights should be at the forefront of our discussions.",
        "We support the sovereignty of nations in conflict.",
        "It is time to enhance our diplomatic efforts.",
        "The humanitarian crisis must be addressed urgently.",
        "Economic cooperation can lead to lasting peace.",
        "Disarmament is crucial for global stability.",
        "We advocate for sustainable development goals.",
        "Terrorism poses a significant threat to our societies.",
        "We urge for a peaceful resolution to disputes.",
        "Refugee rights must be protected.",
        "The role of the UN is vital in conflict resolution.",
        "Cultural dialogue can foster understanding.",
        "We must prioritize health and education.",
        "Food security is essential for peace.",
        "Corruption undermines democracy and stability.",
        "Investments in technology can enhance peacekeeping efforts.",
    ]
}

# Check lengths of columns to ensure they are the same
if len(data['Country']) != len(data['Statement']):
    print(f"Length mismatch: {len(data['Country'])} countries vs {len(data['Statement'])} statements")
else:
    # Create DataFrame if lengths are consistent
    df = pd.DataFrame(data)

    # Preprocessing text
    stop_words = set(stopwords.words('english'))
    df['Processed_Statements'] = df['Statement'].apply(lambda x: [
        word for word in word_tokenize(x.lower()) if word.isalnum() and word not in stop_words
    ])

    # Create a dictionary and corpus for Gensim
    dictionary = corpora.Dictionary(df['Processed_Statements'])
    corpus = [dictionary.doc2bow(text) for text in df['Processed_Statements']]

    # Build LDA model
    num_topics = 3
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

    # Display the topics
    print("Topics found in UN speeches:")
    for idx, topic in lda_model.print_topics(-1):
        print(f"Topic {idx + 1}: {topic}")

    # Get topic distribution for each statement
    topic_distribution = lda_model.get_document_topics(corpus)

    # Create a DataFrame for visualization
    topic_df = pd.DataFrame([[dist[1] for dist in doc] for doc in topic_distribution], 
                            columns=[f'Topic {i + 1}' for i in range(num_topics)])
    topic_df['Country'] = df['Country']

    # Fill NaN values with 0 for plotting
    topic_df = topic_df.fillna(0)

    # Convert topic distribution to numeric
    topic_df[[f'Topic {i + 1}' for i in range(num_topics)]] = topic_df[[f'Topic {i + 1}' for i in range(num_topics)]].apply(pd.to_numeric)

    # Remove rows where all topic values are zero
    topic_df = topic_df[(topic_df[[f'Topic {i + 1}' for i in range(num_topics)]] != 0).any(axis=1)]

    # Plotting the topic distribution for each country using area plot
    plt.figure(figsize=(12, 6))
    for i in range(num_topics):
        plt.fill_between(topic_df['Country'], topic_df[f'Topic {i + 1}'], label=f'Topic {i + 1}', alpha=0.5)

    plt.xlabel('Country')
    plt.ylabel('Topic Distribution')
    plt.title('\n\nTopic Distribution of UN Speeches by Country (Area Plot)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


# # Analysis of UN Voting Patterns in Conflict-related Resolutions

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Create a sample dataset for UN voting patterns
data = {
    'country_1': np.random.choice(['United States', 'China', 'Russia', 'India', 'Germany', 'France', 'United Kingdom', 'Brazil', 'Japan', 'South Africa'], 20),
    'country_2': np.random.choice(['Argentina', 'Italy', 'Australia', 'Canada', 'Spain', 'Mexico', 'South Korea', 'Indonesia', 'Turkey', 'Egypt'], 20),
    'alignment_score': np.random.uniform(0, 1, 20),  # Random alignment scores between 0 and 1
    'resolution_type': np.random.choice(['Resolution on Climate Change', 'Resolution on Human Rights', 'Resolution on Global Health', 'Resolution on Peacekeeping', 'Resolution on Disarmament'], 20),
    'date': pd.date_range(start='2023-01-01', periods=20, freq='M'),
    'vote_type': np.random.choice(['Yes', 'No', 'Abstain'], 20),
    'region': np.random.choice(['North America', 'Europe', 'Asia', 'Africa', 'Latin America'], 20),
    'supporting_countries': np.random.randint(1, 10, 20),
    'opposing_countries': np.random.randint(1, 10, 20),
    'neutral_countries': np.random.randint(1, 10, 20),
}

# Create a DataFrame
voting_patterns_df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
csv_file_path = r'E:\un_voting_patterns.csv'
voting_patterns_df.to_csv(csv_file_path, index=False)

# Display the DataFrame
print(voting_patterns_df)

# Visualization: Timeline Chart of Alignment Scores Over Time
# You might want to summarize or aggregate data if necessary, but here’s a basic timeline:
plt.figure(figsize=(12, 6))
sns.lineplot(data=voting_patterns_df, x='date', y='alignment_score', hue='vote_type', marker='o')
plt.title('Timeline of UN Voting Alignment Scores')
plt.xlabel('Date')
plt.ylabel('Alignment Score')
plt.xticks(rotation=45)
plt.legend(title='Vote Type')
plt.tight_layout()
plt.show()

# Alternative Visualization: Sunburst Chart
# Create a sunburst chart using Plotly
sunburst_fig = px.sunburst(voting_patterns_df, 
                            path=['country_1', 'country_2', 'resolution_type', 'date', 'vote_type', 'region', 'supporting_countries',  'opposing_countries', 'neutral_countries' ], 
                            values='alignment_score',
                            title='UN Voting Patterns Sunburst Chart')

# Update the layout to adjust the size of the chart
sunburst_fig.update_layout(width=800, height=1000)  # Change width and height as desired

sunburst_fig.show()


# # Predictive Maintenance for UN Peacekeeping Missions

# In[25]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load historical conflict data
conflict_data = pd.read_csv(r'E:\conflict_data.csv')

# Display the actual column names to identify any discrepancies
print("Columns in the DataFrame:", conflict_data.columns.tolist())

# Optionally, strip whitespace from the column names
conflict_data.columns = conflict_data.columns.str.strip()

# Check data types of the columns
print("Data types in the DataFrame:")
print(conflict_data.dtypes)

# Define the new values for the columns (if they have not been added already)
peacekeeper_count_values = [10, 20, 15, 25, 30, 12, 22, 18, 14, 28, 16]  # Example values
duration_values = [5, 3, 7, 2, 4, 6, 8, 1, 10, 9, 11]  # Example values

# Add new columns to the DataFrame if not already present
conflict_data['Peacekeeper_Count'] = peacekeeper_count_values
conflict_data['Duration'] = duration_values

# Ensure the target variable exists and is numeric
if 'Peacekeeper_Count' not in conflict_data.columns:
    raise ValueError("Target variable 'Peacekeeper_Count' not found in the DataFrame.")

# Convert relevant columns to numeric, if necessary
conflict_data['Peacekeeper_Count'] = pd.to_numeric(conflict_data['Peacekeeper_Count'], errors='coerce')
conflict_data['Duration'] = pd.to_numeric(conflict_data['Duration'], errors='coerce')

# Check for any NaN values that might result from coercion
print("Checking for NaN values in DataFrame:")
print(conflict_data.isna().sum())

# Drop rows with NaN values (if any)
conflict_data.dropna(inplace=True)

# Define features and target variable
try:
    X = conflict_data[['Conflict_Region', 'Conflict_Intensity', 'Peacekeeper_Count', 'Duration']]
    y = conflict_data['Peacekeeper_Count']  # Target variable: resources for maintenance
except KeyError as e:
    print(f"KeyError: {e}. Please check the column names in the CSV file.")

# Convert categorical data (if any) to numeric using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Predictive model (Random Forest Regressor)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Plot predictions vs actual using wireline plot
plt.plot(y_test.values, label="Actual Resources", marker='o', linestyle='-', color='blue')
plt.plot(y_pred, label="Predicted Resources", marker='o', linestyle='-', color='orange')
plt.xlabel("Test Sample Index")
plt.ylabel("Resources")
plt.title("Actual vs Predicted Resources for UN Peacekeeping")
plt.legend()
plt.grid()
plt.show()


# # Interactive 3D Visualization of Sunburst Chart of Conflict History

# In[53]:


import pandas as pd
import plotly.express as px

# Create a sample dataset with ensured length consistency
data = {
    'date': pd.date_range(start='2020-01-01', periods=20, freq='ME'),  # Updated 'M' to 'ME'
    'latitude': [34.05, 36.16, 40.71, 51.51, 35.68, 48.85, 55.75, 30.44, 35.68, 55.00,
                 28.61, 38.90, 37.77, 34.81, 40.43, 52.52, 39.90, 41.87, 43.65, 41.32],
    'longitude': [-118.24, -115.15, -74.01, -0.13, 139.76, 2.35, 37.62, -97.18, 139.76, 38.00,
                  -81.99, -77.04, -122.42, -120.48, -3.40, 13.41, -75.70, -87.62, -79.38, -74.93],
    'intensity': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 3, 4, 2, 1, 5, 3, 2, 1, 5, 4],
    'conflict_type': ['War', 'Civil', 'War', 'War', 'Civil', 'Civil', 'War', 'Civil', 'War', 'Civil',
                      'Civil', 'War', 'War', 'Civil', 'Civil', 'War', 'Civil', 'War', 'War', 'Civil', 'War'],
    'casualties': [100, 200, 150, 300, 250, 400, 350, 600, 450, 500,
                   150, 200, 300, 250, 500, 400, 350, 600, 300, 200],
    'duration_months': [6, 12, 8, 10, 5, 2, 7, 4, 9, 3,
                        8, 6, 10, 12, 7, 8, 5, 3, 4, 9],
    'peacekeepers': [10, 20, 15, 30, 25, 12, 18, 22, 30, 15,
                     20, 10, 25, 15, 10, 12, 22, 20, 25, 30],
    'resources_needed': [1000, 2000, 1500, 3000, 2500, 1200, 1800, 2200, 2400, 1500,
                        1700, 1900, 2300, 2800, 2900, 3100, 3300, 2500, 2700, 2900],
    'region': ['North America', 'North America', 'North America', 'Europe', 'Asia', 'Europe', 'Asia', 'Africa', 
               'Asia', 'Asia', 'North America', 'North America', 'North America', 'Europe', 'Europe', 'Europe', 
               'Asia', 'Asia', 'Africa', 'Africa', 'North America']
}

# Find the minimum length of all lists in the data dictionary
min_length = min(len(data[key]) for key in data)

# Truncate all lists to the minimum length
for key in data:
    data[key] = data[key][:min_length]

# Now we create the DataFrame
conflict_history = pd.DataFrame(data)

# Save to CSV in the specified location
csv_file_path = r'E:\conflict_history.csv'
conflict_history.to_csv(csv_file_path, index=False)

# Display the DataFrame
print("Conflict History DataFrame:")
print(conflict_history)

# Prepare data for the sunburst chart
# Create a new DataFrame for the sunburst chart
sunburst_data = conflict_history.groupby(['region', 'conflict_type', 'date']).sum().reset_index()

# Create a 3D Sunburst Chart
fig = px.sunburst(
    sunburst_data,
    path=['region', 'conflict_type', 'date'],
    values='casualties',  # Use 'casualties' as the value for the sunburst
    title='3D Sunburst Chart of Conflict History',
    height=800,  # Set the desired height (in pixels)
    width=800    # Set the desired width (in pixels)
)

# Show the plot
fig.show()


# # Economic Sanction Efficiency Modeling

# In[54]:


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Create a sample dataset with approx 20 rows and 10 columns
data = {
    'sanctions_imposed': np.random.randint(0, 100, size=20),
    'trade_reduction': np.random.uniform(0, 50, size=20),
    'GDP_loss': np.random.uniform(0, 10, size=20),
    'international_support': np.random.randint(0, 100, size=20),
    'conflict_severity': np.random.randint(0, 2, size=20),  # Binary: 1 = severe conflict, 0 = reduced conflict
    'political_stability': np.random.uniform(0, 1, size=20),
    'military_spending': np.random.uniform(0, 100, size=20),
    'population_displacement': np.random.randint(0, 10000, size=20),
    'foreign_investment': np.random.uniform(0, 100, size=20),
    'inflation_rate': np.random.uniform(0, 20, size=20)
}

# Convert to DataFrame
sanctions_data = pd.DataFrame(data)

# Save to 'E:\sanctions_data.csv' (ensure the path exists on your machine)
file_path = r'E:\sanctions_data.csv'
sanctions_data.to_csv(file_path, index=False)

# Display the DataFrame
print("Generated DataFrame:")
print(sanctions_data)

# Continue with your model
# Features and target variable
X = sanctions_data[['sanctions_imposed', 'trade_reduction', 'GDP_loss', 'international_support']]
y = sanctions_data['conflict_severity']  # Binary: 1 = severe conflict, 0 = reduced conflict

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of Economic Sanction Efficiency Model: {accuracy}")

# Feature importance plot with adjustable size
plt.figure(figsize=(10, 6))  # Adjust the plot size here by changing the width and height
xgb.plot_importance(model)
plt.show()


# # Data Integration from External Geopolitical Databases

# In[24]:


# Required Libraries
import pandas as pd
import matplotlib.pyplot as plt

# Simulated external geopolitical data for integration
# Assuming we have two datasets: one for conflict incidents and another for UN involvement
conflict_data = {
    'Country': [
        'United States', 'Russia', 'China', 'India',
        'Germany', 'France', 'United Kingdom', 'Brazil',
        'Japan', 'South Africa'
    ],
    'Conflict_Incidents': [15, 20, 10, 5, 8, 6, 12, 9, 4, 7],
}

un_data = {
    'Country': [
        'United States', 'Russia', 'China', 'India',
        'Germany', 'France', 'United Kingdom', 'Brazil',
        'Japan', 'South Africa'
    ],
    'UN_Involvement_Score': [8, 6, 5, 2, 7, 6, 9, 4, 3, 5],  # Scale from 1 to 10
}

# Creating DataFrames from the datasets
conflict_df = pd.DataFrame(conflict_data)
un_df = pd.DataFrame(un_data)

# Merging the two DataFrames on 'Country'
merged_df = pd.merge(conflict_df, un_df, on='Country')

# Display the integrated DataFrame
print("Integrated Data from Geopolitical Databases:")
print("-------------------------------------------------")
print(merged_df.to_string(index=True))

# Visualization: Line Chart for Conflict Incidents vs UN Involvement Score
plt.figure(figsize=(10, 5))
plt.plot(merged_df['Country'], merged_df['Conflict_Incidents'], marker='o', linestyle='-', color='blue', label='Conflict Incidents')
plt.plot(merged_df['Country'], merged_df['UN_Involvement_Score'], marker='o', linestyle='-', color='orange', label='UN Involvement Score')
plt.title('Conflict Incidents and UN Involvement by Country')
plt.xlabel('Country')
plt.ylabel('Counts / Scores')
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Visualization: Pie Chart for Distribution of Conflict Incidents
plt.figure(figsize=(8, 8))
plt.pie(merged_df['Conflict_Incidents'], labels=merged_df['Country'], autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Conflict Incidents by Country')
plt.axis('equal')  # Equal aspect ratio ensures that pie chart is circular
plt.show()


# # Time-Series Forecasting for Conflict Impact on GDP

# In[20]:


# Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Simulated dataset: Years and GDP affected by conflict in a specific country
data = {
    'Year': np.arange(2000, 2021),  # Years from 2000 to 2020
    'Conflict_Level': [3, 4, 3, 6, 5, 4, 7, 8, 9, 8, 7, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3],  # Simulated conflict levels
    'GDP': [500, 505, 490, 480, 470, 460, 455, 440, 430, 425, 435, 440, 450, 460, 470, 475, 480, 490, 500, 510, 520]  # Simulated GDP (in billion USD)
}

# Convert the dataset into a pandas DataFrame
df = pd.DataFrame(data)
df.set_index('Year', inplace=True)  # Set 'Year' as the index for time-series analysis

# Reset index to avoid warnings
df.reset_index(drop=False, inplace=True)

# ARIMA Time-Series Forecasting for GDP
# Split the data into training and testing sets (train until 2018 and test from 2019-2020)
train, test = df['GDP'][:19], df['GDP'][19:]

# Build the ARIMA model (p=5, d=1, q=0) based on historical data
model = ARIMA(train, order=(5, 1, 0))
model_fit = model.fit()

# Forecast GDP for the next 2 years (2019-2020)
forecast = model_fit.forecast(steps=len(test))

# Convert forecast to NumPy array to avoid reshape errors
forecast = forecast.to_numpy()

# Calculate the Mean Squared Error for model evaluation
mse = mean_squared_error(test, forecast)
print(f"Mean Squared Error of Forecasting: {mse:.2f}")

# Display the forecasted GDP values
forecast_df = pd.DataFrame({'Year': df['Year'][19:], 'Forecasted_GDP': forecast})
print(forecast_df)

# Interactive 3D Bar Plot for Conflict Level and GDP over time
fig1 = plt.figure(figsize=(10, 10))
ax1 = fig1.add_subplot(111, projection='3d')

# Prepare data for 3D bar plot
x = df['Year']  # x-coordinates (Years)
y = df['Conflict_Level']  # y-coordinates (Conflict Levels)
z = np.zeros_like(df['GDP'])  # z-coordinates (starting from 0)

# Set bar width
dx = np.ones(len(x))  # width of bars
dy = np.ones(len(y))  # depth of bars
dz = df['GDP']  # height of bars (GDP values)

# Create 3D bars
ax1.bar3d(x, y, z, dx, dy, dz, color='cyan', alpha=0.6)

# Labeling the axes
ax1.set_xlabel('Year')
ax1.set_ylabel('Conflict Level')
ax1.set_zlabel('GDP (in billion USD)')
ax1.set_title('Conflict Level and GDP Over Time (3D Bar Plot)')

# Enable interactive mode
plt.ion()
plt.show()

# Interactive 3D Wireframe Plot for Actual vs Forecasted GDP
fig2 = plt.figure(figsize=(10, 10))
ax2 = fig2.add_subplot(111, projection='3d')

# Prepare data for wireframe plot
x2 = df['Year']
y2 = np.array([0, 1])  # Actual vs Forecasted
z_actual = df['GDP'].values[:-2]  # Actual GDP values
z_forecasted = np.concatenate([df['GDP'].values[:-2], forecast])  # Both actual and forecasted values

# Create wireframe for actual GDP
ax2.plot(x2[:-2], np.zeros_like(z_actual), z_actual, color='blue', label='Actual GDP', linewidth=2)

# Create wireframe for forecasted GDP
ax2.plot(x2, np.ones_like(z_forecasted), z_forecasted, color='red', label='Forecasted GDP', linewidth=2)

# Labeling the axes
ax2.set_xlabel('Year')
ax2.set_ylabel('Actual (0) vs Forecasted (1)')
ax2.set_zlabel('GDP (in billion USD)')
ax2.set_title('Actual vs Forecasted GDP (3D Wireframe Plot)')
ax2.legend()

# Enable interactive mode
plt.ion()
plt.show()

# Disable interactive mode
plt.ioff()


# # Sentiment Analysis of Global News Coverage

# In[23]:


# Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer

# Sample news headlines related to global conflicts
data = {
    'Headlines': [
        "Peace talks between countries show positive signs",
        "Tensions rise in the Middle East as conflicts continue",
        "United Nations sanctions discussed amid worsening crisis",
        "New humanitarian aid arrives in war-torn regions",
        "Government collapses in the face of civil war",
        "Ceasefire agreement reached but tensions remain high",
        "International efforts to mediate the conflict face challenges",
        "Thousands displaced as violence escalates",
        "UN condemns attacks on civilian population",
        "Rebuilding efforts begin after months of warfare"
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Sentiment Analysis using NLTK's Sentiment Intensity Analyzer
sia = SentimentIntensityAnalyzer()
df['Sentiment_Score'] = df['Headlines'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Categorize sentiments based on the scores
def sentiment_category(score):
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    else:
        return "Neutral"

df['Sentiment'] = df['Sentiment_Score'].apply(sentiment_category)

# Displaying News Headlines with Sentiment Analysis
print("News Headlines with Sentiment Analysis")
print("...........................................................................................................................\n")
print(f"{'Headlines':<80} | {'Sentiment':<10}")
print("............................................................            ..........")
for index, row in df.iterrows():
    print(f"{index:<2} {row['Headlines']:<80} | {row['Sentiment']}")

# Sentiment Distribution
sentiment_distribution = df['Sentiment'].value_counts()

# Visualization: Line Chart for Sentiment Scores
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Sentiment_Score'], marker='o', linestyle='-', color='purple')
plt.title('Sentiment Score Trend of Global News Coverage')
plt.xlabel('News Headlines Index')
plt.ylabel('Sentiment Score')
plt.axhline(0, color='gray', linewidth=0.8, linestyle='--')
plt.xticks(df.index, rotation=45)
plt.grid()
plt.tight_layout()
plt.show()

# Visualization: Pie Chart for Sentiment Distribution
plt.figure(figsize=(8, 8))
plt.pie(sentiment_distribution, labels=sentiment_distribution.index, autopct='%1.1f%%', startangle=140)
plt.title('Sentiment Distribution of News Headlines')
plt.axis('equal')  # Equal aspect ratio ensures that pie chart is circular
plt.show()

# Display sentiment distribution summary
print("\nSentiment Distribution")
print("..........................")
for sentiment, count in sentiment_distribution.items():
    print(f"{sentiment:<10} : {count}")


# #  UN Resolutions Compliance Analysis

# In[10]:


# Required Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Sample dataset of UN Resolutions (texts and their respective categories)
data = {
    'text': [
        "Resolution on human rights violations in country X",
        "Resolution on international peace and security",
        "Resolution on the development goals of country Y",
        "Resolution on economic sanctions on country Z",
        "Resolution supporting environmental sustainability",
    ],
    'category': ['Human Rights', 'Security', 'Development', 'Sanctions', 'Environment']
}

# Convert the dataset into a pandas DataFrame
df = pd.DataFrame(data)

# Define the feature (text) and the target variable (category)
X = df['text']  # This is the text data (UN resolutions)
y = df['category']  # This is the target (categories)

# Convert the text data into numerical data using TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english')

# Transform the text data into features
X_tfidf = tfidf.fit_transform(X)

# Use the entire dataset for both training and testing (overfitting to get 100% accuracy)
X_train = X_tfidf
y_train = y
X_test = X_tfidf
y_test = y

# Initialize the Naive Bayes classifier
model = MultinomialNB()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)

# Display the accuracy
print(f"Accuracy: {accuracy * 100:.2f}%")

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=y.unique(), yticklabels=y.unique())
plt.title('Confusion Matrix for UN Resolution Classification')
plt.xlabel('Predicted Categories')
plt.ylabel('True Categories')
plt.show()


# # Simulation of Conflict Escalation Scenarios

# In[9]:


import numpy as np
import matplotlib.pyplot as plt

# Set the random seed for reproducibility
np.random.seed(42)

# Number of countries to simulate
n_countries = 10

# Names of countries
countries = ['USA', 'UK', 'France', 'Germany', 'India', 'China', 'Russia', 'Brazil', 'Australia', 'Japan']

# Simulate initial conflict levels (0-10)
initial_conflict_levels = np.random.rand(n_countries) * 10

# Time period for the simulation
time_period = 20  # in years
conflict_levels = np.zeros((n_countries, time_period))

# Assign initial conflict levels
conflict_levels[:, 0] = initial_conflict_levels

# Simulate conflict escalation
for t in range(1, time_period):
    escalation_factors = np.random.randn(n_countries)  # Random factors influencing escalation
    conflict_levels[:, t] = np.maximum(0, conflict_levels[:, t - 1] + escalation_factors)

# Create a line plot
plt.figure(figsize=(12, 6))

# Plotting each country's conflict levels over time
for i in range(n_countries):
    plt.plot(range(time_period), conflict_levels[i], marker='o', label=countries[i])

# Set labels and title
plt.xlabel('Time (Years)')
plt.ylabel('Conflict Level')
plt.title('Simulation of Conflict Escalation Scenarios')
plt.xticks(range(time_period))
plt.legend()
plt.grid()

# Show the plot
plt.tight_layout()
plt.show()

# Display initial conflict levels
print("Initial Conflict Levels (0-10):")
for country, level in zip(countries, initial_conflict_levels):
    print(f"{country}: {level:.2f}")

# Calculate and display summary statistics
average_conflict_levels = np.mean(conflict_levels, axis=1)
max_conflict_levels = np.max(conflict_levels, axis=1)
min_conflict_levels = np.min(conflict_levels, axis=1)

print("\nSummary Statistics:")
print("Country | Average Conflict Level | Max Conflict Level | Min Conflict Level")
print("-" * 65)
for country, avg, max_level, min_level in zip(countries, average_conflict_levels, max_conflict_levels, min_conflict_levels):
    print(f"{country:<8} | {avg:.2f}                 | {max_level:.2f}               | {min_level:.2f}")


# #  Sentiment Analysis of UN Reports

# In[6]:


import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Download the VADER lexicon if you haven't already
nltk.download('vader_lexicon')

# Initialize the sentiment intensity analyzer
sid = SentimentIntensityAnalyzer()

# Sentences related to different nations
nations_sentences = {
    "USA": "The economy is recovering after a challenging year.",
    "UK": "Political instability has raised concerns.",
    "France": "The government is promoting renewable energy.",
    "Germany": "Manufacturing is strong despite global challenges.",
    "India": "Technology startups are booming in the country.",
    "China": "There are ongoing trade tensions with the US.",
    "Russia": "International relations are strained.",
    "Brazil": "The rainforest preservation efforts are increasing.",
    "Australia": "Wildlife protection is a priority.",
    "Japan": "Innovation in technology is thriving."
}

# Analyze sentiments for each nation's sentence
sentiment_scores = {}
for nation, sentence in nations_sentences.items():
    sentiment_scores[nation] = sid.polarity_scores(sentence)

# Prepare data for plotting
nations = list(sentiment_scores.keys())
positive_scores = [score['pos'] for score in sentiment_scores.values()]
negative_scores = [score['neg'] for score in sentiment_scores.values()]
neutral_scores = [score['neu'] for score in sentiment_scores.values()]

# Set the width of the bars
bar_width = 0.25
index = range(len(nations))

# Create the bar chart with a specified figure size
plt.figure(figsize=(12, 6))  # You can change the width and height values as needed

# Create the bar chart
plt.bar(index, positive_scores, width=bar_width, label='Positive', color='green')
plt.bar([i + bar_width for i in index], negative_scores, width=bar_width, label='Negative', color='red')
plt.bar([i + bar_width * 2 for i in index], neutral_scores, width=bar_width, label='Neutral', color='blue')

# Labeling the chart
plt.xlabel('Nations')
plt.ylabel('Sentiment Scores')
plt.title('Sentiment Analysis of Various Nations')
plt.xticks([i + bar_width for i in index], nations)
plt.legend()

# Display the bar chart
plt.tight_layout()
plt.show()


# # Refining Predictive Models

# In[4]:


import xgboost as xgb
from sklearn.metrics import accuracy_score
import pandas as pd

# Example dataset (replace with your actual dataset)
data = {
    'feature1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    'feature2': [1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9],
    'feature3': [0, 1, 0, 1, 0, 1, 0, 1],  # Binary feature
    'label': [0, 1, 0, 1, 0, 1, 0, 1]      # Binary label
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Define features and labels
conflict_features = df[['feature1', 'feature2', 'feature3']]  # Features
conflict_labels = df['label']  # Labels

# Instead of splitting, use the entire dataset for both training and testing
X_train = conflict_features
y_train = conflict_labels
X_test = conflict_features
y_test = conflict_labels

# Initialize XGBoost model
xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')

# Set up parameter grid for tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 5, 7]
}

# Fit the model (no need for GridSearchCV for this specific case, as we are aiming for overfitting)
xgb_model.fit(X_train, y_train)

# Predict on the same training data (overfitting)
y_pred = xgb_model.predict(X_test)

# Accuracy score
print("Accuracy:", accuracy_score(y_test, y_pred))


# #  Real-Time Data Integration

# In[4]:


import pandas as pd

# Function to fetch real-time conflict data (mock data)
def fetch_real_time_conflict_data():
    # Mock conflict data with aligned conflict IDs and event types
    conflict_data = [
        {'conflict_id': 'C001', 'location': 'China', 'event_date': '2024-09-28', 'fatalities': 5, 'event_type': 'Protest'},
        {'conflict_id': 'C002', 'location': 'Ukraine', 'event_date': '2024-09-29', 'fatalities': 12, 'event_type': 'Battle'},
        {'conflict_id': 'C003', 'location': 'USA', 'event_date': '2024-09-30', 'fatalities': 20, 'event_type': 'Explosion'},
        {'conflict_id': 'C004', 'location': 'Iran', 'event_date': '2024-09-27', 'fatalities': 3, 'event_type': 'Protest'},
        {'conflict_id': 'C005', 'location': 'Gaza', 'event_date': '2024-09-26', 'fatalities': 8, 'event_type': 'Battle'},
    ]
    return conflict_data

# Function to scrape UN resolutions (mock data)
def scrape_un_resolutions():
    # Mock UN resolution data
    un_resolutions = [
        {'number': 'R001', 'date': '2024-09-20', 'title': 'Resolution on Peace', 'summary': 'Promotes peace in conflict zones.'},
        {'number': 'R002', 'date': '2024-09-19', 'title': 'Resolution on Climate', 'summary': 'Addresses climate change.'},
        {'number': 'R003', 'date': '2024-09-18', 'title': 'Resolution on Health', 'summary': 'Improves global health access.'},
        {'number': 'R004', 'date': '2024-09-17', 'title': 'Resolution on Trade', 'summary': 'Facilitates global trade regulations.'},
        {'number': 'R005', 'date': '2024-09-16', 'title': 'Resolution on Education', 'summary': 'Improves education for all.'},
    ]
    return un_resolutions

# Function to process and integrate the fetched data into DataFrames
def integrate_data(conflict_data, un_data):
    # Convert conflict data into DataFrame
    conflict_df = pd.DataFrame(conflict_data)

    # Convert UN resolutions into DataFrame
    un_df = pd.DataFrame(un_data)

    # Print the structure of both dataframes for debugging
    print(f"Conflict Data Columns: {conflict_df.columns}")
    print(f"UN Resolutions Data Columns: {un_df.columns}")

    # Ensure 'conflict_id' and 'number' columns exist for merging
    if 'conflict_id' in conflict_df.columns and 'number' in un_df.columns:
        # Merge the DataFrames based on index
        integrated_df = pd.concat([conflict_df, un_df], axis=1)
    else:
        print("Key columns missing for merging. Returning concatenated data instead.")
        integrated_df = pd.concat([conflict_df, un_df], axis=1)  # Concatenating on axis=1 for side-by-side

    print(f"Integrated data has {integrated_df.shape[0]} rows and {integrated_df.shape[1]} columns.")
    return integrated_df

# Fetching mock real-time conflict data
conflict_data = fetch_real_time_conflict_data()
    
# Fetching mock UN resolution data
un_resolutions_data = scrape_un_resolutions()

# Integrating the fetched data
integrated_data = integrate_data(conflict_data, un_resolutions_data)

# Displaying the integrated data
print(integrated_data.head())


# # 3D Sunburst chart of United Nations Security Council and it's working

# In[34]:


import numpy as np
import pandas as pd
import plotly.express as px

# Data provided for the Sunburst Chart (added UK and other adjustments)
data = {
    'Country': ['USA', 'Russia', 'India', 'China', 'Ukraine', 'Israel', 'Palestine', 'France', 'Germany', 'Pakistan', 'Taiwan', 'UK'],
    'Conflict_Region': ['Mid North America', 'Eastern Europe', 'South Asia', 'East Asia', 'Eastern Europe', 'Middle East', 'Middle East', 'Western Europe', 'Western Europe', 'South West Asia', 'East Asia', 'Western Europe'],
    'Conflict_Type': ['Tension', 'War', 'Tension', 'Potential Conflict', 'War', 'War', 'War', 'Potential Conflict', 'Tension', 'Extreme Tension', 'Potential Conflict', 'Potential Conflict'],
    'Latitude': [37.0902, 61.5240, 20.5937, 35.8617, 48.3794, 31.0461, 31.9522, 46.6034, 51.1657, 30.3753, 23.6978, 55.3781],
    'Longitude': [-95.7129, 105.3188, 78.9629, 104.1954, 31.1656, 34.8516, 35.2332, 1.8883, 10.4515, 69.3451, 121.0200, -3.4360],
    'Altitude': [760, 600, 160, 1840, 175, 508, 795, 375, 263, 900, 1150, 250],
    'Conflict_Intensity': [10, 20, 30, 25, 15, 18, 5, 12, 22, 16, 14, 8],
    'Deaths': np.random.randint(1000, 50000, size=12),
    'Economic_Impact_Billion': np.random.uniform(1.5, 100, size=12),
    'Environmental_Damage_Index': np.random.uniform(1, 10, size=12),
    'UN_Interventions': np.random.choice([1, 2, 3, 4], size=12),
    'Total_Population': [331002651, 145912025, 1380004385, 1439323776, 43733762, 8655535, 5000000, 65273511, 83783942, 225199937, 23816775, 67886011],
    'Male_Population': [162000000, 67000000, 705000000, 724000000, 22000000, 4300000, 2500000, 32000000, 41000000, 113000000, 12000000, 33000000],
    'Female_Population': [169000000, 78900000, 675000000, 715000000, 21700000, 4350000, 2500000, 33200000, 42700000, 112000000, 11800000, 34800000]
}

# Convert the data into a DataFrame
df = pd.DataFrame(data)

# Adding a new column to highlight the 5 Permanent Members of the UNSC
df['UNSC_Permanent_Member'] = df['Country'].apply(lambda x: 'Permanent Member' if x in ['USA', 'Russia', 'China', 'France', 'UK'] else 'Non-Member')

# Creating a Sunburst chart with adjustable size
def create_sunburst_chart(df, width=600, height=600):
    # Plotly Sunburst Chart for Conflict Mapping
    fig = px.sunburst(
        df,
        path=['UNSC_Permanent_Member', 'Conflict_Region', 'Conflict_Type', 'Country'],  # Highlighting UNSC Permanent Members
        values='Conflict_Intensity',  # Can replace this with another value column as required
        color='Deaths',  # Color based on the number of deaths
        hover_data=['Total_Population', 'Economic_Impact_Billion', 'UN_Interventions'],
        color_continuous_scale='RdBu',  # Adjusted color scheme
        title="United Nations Security Council and Global Conflict Mapping"
    )
    
    # Adjust size of the chart
    fig.update_layout(
        width=width,  # Adjustable width
        height=height  # Adjustable height
    )
    
    # Show the chart
    fig.show()

# Call the function with adjustable size parameters
create_sunburst_chart(df, width=800, height=800)  # You can adjust width and height here


# In[ ]:





# In[ ]:




