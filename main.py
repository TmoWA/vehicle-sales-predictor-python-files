import pandas as pd  # Imported to read the CSV file.
import matplotlib.pyplot as plt  # Imported to plot the graphs.
import numpy as np  # Imported for numerical needs.
from sklearn import metrics  # Imported to explain the model's performance.
from sklearn.model_selection import train_test_split  # Imported to prepare the training and test data.
from sklearn.ensemble import RandomForestRegressor  # RandomForestRegressor will be the model used.

# Dataset acquired from:
# https://catalog.data.gov/dataset/mva-vehicle-sales-counts-by-month-for-calendar-year-2002-2020-up-to-october
# Please note that the URL states the incorrect time range.

csv_file = 'mva_sales_data.csv'  # Name of the data file to be used.
new_sales_regressor = None  # Variable to hold the model that will predict new automobile sales.
used_sales_regressor = None  # Same as the line above but for used automobile sales.

# This code will show the original graphs for the sales data. It includes both used and new sales.
headers = ['Year', 'Month', 'New', 'Used', 'Total New Sales', 'Total Used Sales']
df = pd.read_csv(csv_file, names=headers, skiprows=1)  # The skip is for the headers row.
plt.scatter(x=df.index, y=df['New'], c='Red')
plt.scatter(x=df.index, y=df['Used'], c='Blue')
plt.xlabel('Row Number')
plt.ylabel('Sales Quantity')

plt.title('Total Sales Over Time')
plt.show()

headers = ['Year', 'Month', 'New', 'Used', 'Total New Sales', 'Total Used Sales']
df = pd.read_csv(csv_file, names=headers, skiprows=1)  # The skip is for the headers row.
plt.scatter(x=df['Year'], y=df['New'], c='Red')
plt.scatter(x=df['Year'], y=df['Used'], c='Blue')
plt.xlabel('Year')
plt.ylabel('Sales Quantity')

plt.title('Sales by Year')
plt.show()

headers = ['Year', 'Month', 'New', 'Used', 'Total New Sales', 'Total Used Sales']
df = pd.read_csv(csv_file, names=headers, skiprows=1)  # The skip is for the headers row.
plt.scatter(x=df['Month'], y=df['New'], c='Red')
plt.scatter(x=df['Month'], y=df['Used'], c='Blue')
plt.xlabel('Month Number')
plt.ylabel('Sales Quantity')

plt.title('Sales by Month')
plt.show()


def evaluate(y_test, y_pred):
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    print('\tRoot Mean Squared Error:', rmse)
    normalized_rmse = rmse / (y_test.max() - y_test.min())
    print('\tNormalized RMSE:', normalized_rmse)


def rfr_model(name, color):
    rfr_headers = ['Year', 'Month', 'New', 'Used', 'Total New Sales', 'Total Used Sales']  # Column names
    rfr_df = pd.read_csv(csv_file, names=rfr_headers, skiprows=1)  # The skip is for the headers row.
    x = rfr_df['Month'].to_numpy().reshape(-1, 1)  # Converts the values into an appropriate array for the model.
    y = rfr_df[name].to_numpy()  # Similar to the line above but for the column name passed to the function.

    # The following divides the data into training and testing data.
    # It performs a 70/30 split for the training data and the testing data, respectively.
    # We will be also setting a random state for this Notebook to keep results consistent.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=25)

    # We can now instantiate the Random Forest Regressor (RFR) model and begin fitting it with the training data.
    # Additionally, for this Notebook, we will use a state to keep the regressors consistent across demonstrations.
    regressor = RandomForestRegressor(n_estimators=300, random_state=25)
    regressor.fit(x_train, y_train)

    # Plots the test data for display using the color passed to this function.
    plt.scatter(x_test, y_test, c=color)

    # Plots the predictions over the test data graph. The predictions will be in green.
    x_grid = np.arange(x.min(), x.max(), 1)
    x_grid = x_grid.reshape((len(x_grid)), 1)
    plt.plot(x_grid, regressor.predict(x_grid), color='Green')
    plt.xlabel("Month Number")
    plt.ylabel("Sales Quantity")

    # Calculates an evaluation metric for the tree. As a reminder, we want our RMSE value to be low,
    # and our normalized RMSE to be close to zero.
    y_pred = regressor.predict(x_test)  # Retrieves some predictions based on the test data.
    evaluate(y_test, y_pred)

    plt.title('Sales Predictions')
    plt.show()
    return regressor


new_sales_regressor = rfr_model('New', 'Red')  # Generates a model for the "new sales" data and displays it in red.
used_sales_regressor = rfr_model('Used', 'Blue')  # Generates a model for the "used sales" data and displays it in blue.

headers = ['Year', 'Month', 'New', 'Used', 'Total New Sales', 'Total Used Sales']
df = pd.read_csv(csv_file, names=headers, skiprows=1)  # The skip is for the headers row.
sums = [df['New'].sum(), df['Used'].sum()]  # Sums the values for each column.
values = np.array(sums)
plt.pie(values, labels=['New Automobile Sales', 'Used Automobile Sales'],
        colors=['Red', 'Blue'], autopct='%1.2f%%')
plt.title('Automobile Sales by Type')
plt.show()

headers = ['Year', 'Month', 'New', 'Used']
df = pd.read_csv('sales_subset_2023.csv', names=headers,
                 skiprows=1)  # The skip is for the headers row.
print(df)

for month in range(1, 13):
    # The first number is the start and the second is the exclusive ending point, meaning this is a range of 1 to 12.
    new_sales_prediction = new_sales_regressor.predict([[month]])
    used_sales_prediction = used_sales_regressor.predict([[month]])
    print('Prediction for month', month, '- New:', new_sales_prediction, '- Used:', used_sales_prediction)
