The dataset comes from https://www.kaggle.com/datasets/zubairamuti/shopping-behaviours-dataset/data



Perform simple single metric analysis on ipynb files
###1. * * Customer demographic analysis**

-Age distribution analysis (age group preference)
-Gender differences in purchasing behavior
-Cross analysis of age and gender
-Analysis of Consumption Ability in Different Age Groups

###2. * * Product Category and Preference Analysis**

-Sales distribution of product categories
-Gender preferences for product categories
-Seasonal commodity purchasing trend
-Distribution pattern of product size
-Color preference analysis

###3. Analysis of Consumer Behavior**

-Distribution of purchase amount
-Average consumption levels of different age groups
-The relationship between gender and consumption amount
-The correlation between historical purchase frequency and current consumption

###4. * * Geographic analysis**

-Heat map of sales distribution by state
-Differences in regional consumption preferences
-Average consumption levels in different regions

###5. Seasonal analysis**

-Trend of Four Seasons Sales Changes
-The relationship between seasons and product categories
-Seasonal color preference
-The impact of seasons on consumption amounts

###6. Customer Satisfaction Analysis**

-Distribution of evaluation scores
-The relationship between rating and product category
-The correlation between rating and consumption amount
-Customer satisfaction with different purchase frequencies

###7. Analysis of Marketing Strategy Effectiveness**

-Discount usage and its impact on sales
-Analysis of Promotion Code Usage
-Differences in behavior between subscribed and non subscribed customers
-Preference for using different payment methods

###8. Logistics distribution analysis**

-Preferred delivery method selection
-The relationship between delivery method and consumption amount
-Selection of delivery methods in different regions

###9. Customer loyalty analysis**

-Purchase frequency distribution
-Analysis of historical purchase frequency
-Characteristic portrait of loyal customers
-The relationship between purchase frequency and consumption amount

###10. * * Comprehensive correlation analysis**

-Multidimensional cross analysis (age gender category season)
-Clustering of customer behavior patterns
-Construction of Consumer Profile
-Correlation analysis of product recommendations

Python file utilizes streamlit for further indicator correlation analysis

geo_advanced_dashboard.py
Advanced Geographic Visualization Analysis Dashboard.
Support filtering data by gender, category, and season, dynamically displaying sales heat maps of various states in the United States.
Implement regional customer clustering distribution (KMeans), state-level portrait drilling, product recommendation network flow chart (NetworkX), and multi-dimensional radar comparison.
Highlights: Geographic distribution, clustering, portrait pop ups, product flow network, radar charts, suitable for spatial analysis and regional insights.

interactive_dashboard.py
Interactive shopping behavior analysis dashboard.
Support multi-dimensional filtering (gender, age, category, season), drag the timeline to observe trends.
Dynamically display sales trends, age category heat maps, customer clustering dimensionality reduction (KMeans+TSNE), product recommendation relevance network diagrams, and typical consumer portrait cards.
Highlights: Multidimensional interaction, clustering dimensionality reduction, network recommendation, portrait display, YAML configuration, suitable for global behavior analysis and personalized exploration.

rfm_lifecycle_test.py
Customer lifecycle and churn prediction analysis.
Based on the RFM model, stratify customer value, simulate recent purchases, frequency, and amount, and score by group.
Visualize RFM distribution, construct churn labels, train a random forest model to predict churn probability, and display a list of high-risk customers.
Highlights: RFM layering, churn probability prediction, interactive visualization, suitable for customer value assessment and churn warning.


Running code:
pip install -r requirements.txt
streamlit run rfm_lifecycle_test.py  
streamlit run interactive_dashboard.py
streamlit run geo_advanced_dashboard.py
