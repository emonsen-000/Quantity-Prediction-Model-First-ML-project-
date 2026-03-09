import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("Amazon.csv")

### DATA PREPARATION & FEATURED ENGINEERING

# Change OrderDate to datetime
print("Data type before changing to datetime:",df["OrderDate"].dtype)      
df["OrderDate"]=pd.to_datetime(df["OrderDate"])
print("\nData type after changing to datetime:",df["OrderDate"].dtype)   

# changing the state name form
state_map = {
    "TX": "Texas",
    "CA": "California",
    "NC": "North Carolina",
    "FL": "Florida",
    "WA": "Washington",
    "OH": "Ohio",
    "DC": "District of Columbia",
    "IL": "Illinois",
    "PA": "Pennsylvania",
    "CO": "Colorado",
    "IN": "Indiana",
    "NY": "New York",
    "AZ": "Arizona"
}
df["State"]=df["State"].replace(state_map)
print("\nafter changing state name:\n",df["State"])

# adding year and month column

df["Year"]=df["OrderDate"].dt.year
df["Month"]=df["OrderDate"].dt.month

# REVENUE & PROFIT
revenue=df["TotalAmount"].sum()
print("\nTotal Revenue:\n",revenue)

total_shipping_cost=df["ShippingCost"].sum()
profit=revenue-total_shipping_cost
print("\nTotal PROFIT:\n",profit)

# REVENUE & PROFIT (CATEGORY WISE)
rev_cat_wise=df.groupby("Category")["TotalAmount"].sum()
print("\nCategory wise Revenue:\n",rev_cat_wise.sort_values(ascending=False))

cost_cat_wise=df.groupby("Category")["ShippingCost"].sum()
print("\nCategory wise Cost:\n",cost_cat_wise.sort_values(ascending=False))

profit_cat_wise=rev_cat_wise-cost_cat_wise
print("\nCategory wise profit:\n",profit_cat_wise.sort_values(ascending=False))

# Monthly Revenue
monthly_revenue=df.groupby("Month")["TotalAmount"].sum()
print("\nMonthly Revenue:\n",monthly_revenue.sort_index())
monthly_cost=df.groupby("Month")["ShippingCost"].sum()
print("\nMonthly Revenue:\n",monthly_cost.sort_index())

# Extracting state names where country is not Australia or U.S.
df.loc[~df["Country"].isin(["Australia","United States"]),"State"]="Not Given"

### PRODUCT & CATEGORY ANALYSIS

# TOTAL TYPE OF PRODUCTS
print("\nTotal type of products:",df["ProductName"].nunique())
# TOTAL TYPE OF CATEGORIES
print("\nTotal type of categories:",df["Category"].nunique())
# TOTAL QUANTITY SOLD
print("\nTotal quantity sold:",df["Quantity"].sum())

# TOP 10 MOST SOLD PRODUCT
most_sold_product=df.groupby("ProductName")["Quantity"].sum()
sorted_products=most_sold_product.sort_values(ascending=False)
print("\nTop 10 Most Sold Product:\n",sorted_products.head(10))

# NO1 CATEGORY (according to how many product sold)
top_category=df.groupby("Category")["Quantity"].sum()
sorted_category=top_category.sort_values(ascending=False)
print("\nNo 1 Category:\n",sorted_category.head(1))

# Electronics category generates the highest revenue, indicating strong demand.

# Top 10 Most Sold Product In Electronics Category
cat_elec=df[df["Category"]=="Electronics"]
most_in_elect=cat_elec.groupby("ProductName")["Quantity"].sum()
print("\nTop 10 Most sold product in Electronics:\n",most_in_elect.sort_values(ascending=False).head(10))

top_brand_in_elect=cat_elec.groupby("Brand")["Quantity"].sum()
print("\nTop 5 Brand in Electronics:\n",top_brand_in_elect.sort_values(ascending=False).head(5))

# UrbanStyle RANKED 1...

# Top Smartwatch Brand
watches=df[df["ProductName"]=="Smartwatch"]
top_smartwatch=watches.groupby("Brand")["Quantity"].sum()
print("\nTop Smartwatch Brand:\n",top_smartwatch.sort_values(ascending=False).head(1))

# HomeEase sold the most smartwatches. They are not in top 5, but people are liking there Smart watches..

### COST, QUANTITY, TAX & DELIVERY ANALYSIS

total_tax=df["Tax"].sum()
print("\nTotal amount of tax:\n",total_tax)

print("\nTotal shipping cost:\n",total_shipping_cost)

# any quantity<=0?
check_nan=df["Quantity"].isnull().sum()
print("No of missing value in quantity:",check_nan)
#negative value
check_neg=df[df["Quantity"]<0]
print("\nNumber of negative values:\n",check_neg)

# how many deliveries are pending, delivered, shipped, canceled, returned
print("\nDelivery Results:\n",df["OrderStatus"].value_counts())

# TAX mismatch?
gross = df["UnitPrice"] * df["Quantity"]
discounted = gross - (gross * df["Discount"])
approx_tax_rate = df["Tax"] / discounted
print("\nCheck for mismatch:\n",approx_tax_rate.describe())          # no mismatch

# which payment method used most
most_used_method=df["PaymentMethod"].value_counts().sort_values(ascending=False)
print("\nMost used payment methods:\n",most_used_method.head(1))   # CREDIT CARDS are most used method by the customers

### Country & Region Insights
top_country=df["Country"].value_counts()
print("\nTop Country:\n",top_country.head(1))

# Orders from U.S. are the most. Almost 5X more than 2nd ranked India.

# Which country bought most Electronics
high_in_elect=df.groupby("Category")["Country"].value_counts()
print("\nEach categories record of orders from countries:\n",high_in_elect)
print("\nOrders from Countries in Electronics Category:\n",high_in_elect["Electronics"])

# U.S. ordered the most electronic products.

# top categories in each country
high_in_cat=df.groupby("Country")["Category"].value_counts()
print("\nTop categories in each country\n",high_in_cat)
print("\nTop categories in U.S.:\n",high_in_cat["United States"])    # Orders on Books are most in U.S.

# top products in each country
high_in_prod=df.groupby("Country")["ProductName"].value_counts()
print("\nTop Products in each country\n",high_in_prod)
print("\nTop Products in U.S.:\n",high_in_prod["United States"].head(10))

# profit & revenue according to countries
rev_country_wise=df.groupby("Country")["TotalAmount"].sum()
print("\nCountry wise Revenue:\n",rev_country_wise.sort_values(ascending=False))

cost_country_wise=df.groupby("Country")["ShippingCost"].sum()
print("\nCountry wise Cost:\n",cost_country_wise.sort_values(ascending=False))

profit_country_wise=rev_country_wise-cost_country_wise
print("\nCountry wise profit:\n",profit_country_wise.sort_values(ascending=False))

# Most order from which state of U.S.
most_order_in_state=df.groupby("Country")["State"].value_counts()
most_order_in_US=most_order_in_state["United States"]
print("\nOrders from states of U.S.:\n",most_order_in_US)

### LINEAR REGRESSION MODEL

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df_label=df.copy()
le=LabelEncoder()
df_label["Brand Encoder"]=le.fit_transform(df_label["Brand"])
df_label["Category Encoder"]=le.fit_transform(df_label["Category"])
df_label["Payment method Encoder"]=le.fit_transform(df_label["PaymentMethod"])
df_label["Country Encoder"]=le.fit_transform(df_label["Country"])

features=["UnitPrice","Discount","Brand Encoder","Category Encoder","Tax","ShippingCost","Payment method Encoder","Country Encoder","Month"]
X=df_label[features]
y=df_label["Quantity"]

X_train, X_test, Y_train, Y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(X_train,Y_train)
y_pred=model.predict(X_test)
mae=mean_absolute_error(Y_test,y_pred)
mse=mean_squared_error(Y_test,y_pred)
r2=r2_score(Y_test,y_pred)
print("The quantity prediction is:",y_pred)
print("Intercept:",model.intercept_)
print("Coefficient:")
for feat,coef in zip(features,model.coef_):
    print(f"{feat}:{coef:.2f}")
print("MAE",mae.__round__(2))
print("MSE",mse.__round__(2))
print("r2",r2.__round__(3))

plt.scatter(Y_test,y_pred,color="green",alpha=0.4)
plt.plot([Y_test.min(),Y_test.max()],[Y_test.min(),Y_test.max()],color="red")
plt.show()