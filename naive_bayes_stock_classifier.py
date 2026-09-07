

# Name: Saravenus A Khon
# Date: 09/01/2026
# Project - CTA06 - Fun with Naive Bayes Classifier --

#Part 1: Import Libraries
# From scikit-learn's Naive Bayes toolbox -- Give me the Bernoulli classifiers.
import pandas as pd
from pathlib import Path
from sklearn.naive_bayes import BernoulliNB





# Part 2: Program constants and File Locations - Store the stock data and settings used throughout the program
# Constants
STOCK_TICKER = "CCL"                                  # Our stock is Carnival
COMPANY_NAME = "Carnival Corporation & Plc"
PREDICTION_DAYS = 5                                   # We're predicting if the stock will be higher 5 trading days later
MOVING_AVERAGE_DAYS = 20                              # Use this to compare today's price & volume against their 20- trading day averages

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "CCL_HistoricalData_5years.csv"




#Part 3: Load Historical stock data
def load_stock_data():
    # This line creates a Pandas DataFrame.
    stock_data = pd.read_csv(DATA_FILE)     # enables Pandas to read the csv file in the following path
    print("***" * 40)


    print("\nHISTORICAL STOCK DATA")
    print("***" * 40)

    # Print the rows and columns
    print(f"ROWS loaded: {len(stock_data)}")      # uses len () -Counts how many rows are in stock_data and inserts the count into the printed string
    print(f"COLUMNS: {list(stock_data.columns)}") # uses List () -Gets the column names from stock_data, converts them into a list, and inserts the list into the printed string

    print("\nFirst 5 rows: ")
    print(stock_data.head().to_string(index=False, col_space=20))

    return stock_data     # Run the function, get the stock spreadsheet back, and save it as stock_data.






#Part 4- Clean and prep the stock data - This part is necessary because the CSV wasn't immediately ready for calculations
def prepare_stock_data(stock_data):
    # Make a working copy of stock data.
    stock_data = stock_data.copy()

    # convert the Data Column from text into actual dates that Python can understand.
    stock_data["Date"] = pd.to_datetime(stock_data["Date"])

    # Make a list of the four columns that have dollar amounts.
    price_columns = ["Close/Last", "Open", "High", "Low"]

    # go through each one of those four columns one at a time and do the following
    for column in price_columns:
        stock_data[column] = (
            stock_data[column]
            .astype(str)
            .str.replace("$", "", regex=False)        # changes $23.23 to ---> 23.23 to remove the $
            .str.replace(",", "", regex=False)
            .astype(float)                           # Converts text to a decimal number so Python can use this data for math
        )

    # Make sure Volume is stored as a number
    stock_data["Volume"] = pd.to_numeric(stock_data["Volume"], errors="coerce")


    # Put the stock data into chronological order of oldest to most recent date
    stock_data = stock_data.sort_values("Date").reset_index(drop=True)





    # part 5: Creating what we're trying to predict - Make the prediction Target here ---------------------------------
    # compare today's closing price to the closing price 5 trading days later
    #1 = stock increase       # 0 = Stock didn't increase

    # Take the closing price from 5 rows later and put it besides today's row.
    stock_data["Future_Close"] = stock_data["Close/Last"].shift(-PREDICTION_DAYS)

    #Remove the last 5 rows because they don't have future closing price
    stock_data = stock_data.dropna(subset=["Future_Close"])

    # make the class that the Naive Bayes model will predict here ------------------------------------------------------
    stock_data["Higher_In_5_Days"] = (
        stock_data["Future_Close"] > stock_data["Close/Last"]
    ).astype(int)


    # Part 6: Create the features that Bernoulli Naive Bayes will use to make its prediction----------------------------
    # This part gives us 2 binary predictors and the target class


    # Calculate the 20-trading-day average closing price
    stock_data["Price_20_Day_Avg"] = (
        stock_data["Close/Last"].rolling(MOVING_AVERAGE_DAYS).mean()
    )

    # Calculate the 20-trading-day average trading volume
    stock_data["Volume_20_Day_Avg"] = (
        stock_data["Volume"].rolling(MOVING_AVERAGE_DAYS).mean()
    )


    # Create a binary feature for price
    # 1 = today's closing price is above the 20-day average
    # 0 = today's closing price is not above the 20-day average
    stock_data["Price_Above_Avg"] = (
            stock_data["Close/Last"] > stock_data["Price_20_Day_Avg"]
    ).astype(int)



    # Create a binary feature for volume
    # 1 = today's volume is above the 20-day average
    # 0 = today's volume is not above the 20-day average
    stock_data["Volume_Above_Avg"] = (
            stock_data["Volume"] > stock_data["Volume_20_Day_Avg"]
    ).astype(int)



    # Remove rows that do not yet have a full 20-day moving average
    stock_data = stock_data.dropna(
        subset=["Price_20_Day_Avg", "Volume_20_Day_Avg"]
    )

    return stock_data



# Part 7: Create Frequency Tables --------------------------------------------------------------------------------------
# Count how many times each feature value occurs for each prediction class

def create_frequency_tables(stock_data):

    # This the frequency table for price
    # This part of the func asks how many times did each Price_Above_Avg value occur w/ each Higher_in_5_Days result
    price_freq = pd.crosstab(
        stock_data["Price_Above_Avg"],
        stock_data["Higher_In_5_Days"]
    )

    # This is the frequency table for volume
    volume_freq = pd.crosstab(
        stock_data["Volume_Above_Avg"],
        stock_data["Higher_In_5_Days"]
    )

    # Giving the rows and columns easier names for better understanding
    price_display = price_freq.rename(
        index={0: "Not Above Average", 1: "Above Average"},
        columns={0: "Did NOT Increase", 1: "Did Increase"}
    )

    volume_display = volume_freq.rename(
        index={0: "Not Above Average", 1: "Above Average"},
        columns={0: "Did NOT Increase", 1: "Did Increase"}
    )

    print("\nPRICE FREQUENCY TABLE")
    print("-" * 80)
    print(price_display.to_string())
    print("-" * 80)

    print("\nVOLUME FREQUENCY TABLE")
    print("-" * 80)
    print(volume_display.to_string())
    print("-" * 80)


    return price_freq, volume_freq





# Part 8: Create Likelihood Tables -------------------------------------------------------------------------------------
# Convert the frequency counts into probabilities for each prediction class

def create_likelihood_tables(price_freq, volume_freq):

    # Add 1 to each frequency to prevent zero probabilities
    # Add 2 to the total because each feature has two possible values: 0 or 1
    price_likelihood = (price_freq + 1).div(price_freq.sum(axis=0) + 2, axis=1)
    volume_likelihood = (volume_freq + 1).div(volume_freq.sum(axis=0) + 2, axis=1)


    # adding coloumns and rows
    price_display = price_likelihood.rename(
        index={0: "Not Above Average", 1: "Above Average"},
        columns={0: "Did NOT Increase", 1: "Did Increase"}
    )

    volume_display = volume_likelihood.rename(
        index={0: "Not Above Average", 1: "Above Average"},
        columns={0: "Did NOT Increase", 1: "Did Increase"}
    )

    print("\nPRICE LIKELIHOOD TABLE")
    print("-" * 60)
    print(price_display.map(lambda x: f"{x:.2%}").to_string())
    print("-" * 60)

    print("\nVOLUME LIKELIHOOD TABLE")
    print("-" * 60)
    print(volume_display.map(lambda x: f"{x:.2%}").to_string())
    print("-" * 60)


    return price_likelihood, volume_likelihood





# Part 9: Train the Naive Bayes classifier and calculate posterior probabilities----------------------------------------

def run_classifier(stock_data):

    # These are the two features the classifier will use
    X = stock_data[["Price_Above_Avg", "Volume_Above_Avg"]]

    # This is what the classifier is trying to predict
    y = stock_data["Higher_In_5_Days"]

    # alpha=1.0 applies Laplace correction to prevent zero probabilities
    model = BernoulliNB(alpha=1.0)

    # Train the classifier
    model.fit(X, y)


    # Test: 1 = price above average, 1 = volume above average
    # This part is asking the model : suppose today's stock price is over its 20-day average AND today's volume is
    # also above its 20- ay average. What do you think will happen 5 trading days later?
    test_data = pd.DataFrame(
        [[1, 1]],
        columns=["Price_Above_Avg", "Volume_Above_Avg"]
    )


    # Calculate the posterior probability for each class
    probabilities = model.predict_proba(test_data)[0]

    # Make the final class prediction here
    prediction = model.predict(test_data)[0]

    print("\nPOSTERIOR PROBABILITIES")
    print("Test Input: Price Above Average = 1, Volume Above Average = 1")
    print(f"Class 0 - Stock will NOT increase: {probabilities[0]:.2%}")
    print(f"Class 1 - Stock WILL increase: {probabilities[1]:.2%}")

    if prediction == 1:
        print("Prediction: Stock WILL increase.")
    else:
        print("Prediction: Stock will NOT increase.")


    return model





# this is the header
def display_header():
    print("=" * 40)
    print("--- NAIVE BAYES STOCK MARKET CLASSIFIER ---")
    print("=" * 40)
    print(f"Company: {COMPANY_NAME}")
    print(f"Ticker: {STOCK_TICKER}")
    print(f"Prediction Horizon: {PREDICTION_DAYS} trading days")
    print("=" * 40)




# Calls the function
display_header()

# function call to load historical stock data of Carnival
stock_data = load_stock_data()

stock_data = prepare_stock_data(stock_data)

price_frequency, volume_frequency = create_frequency_tables(stock_data)


price_likelihood, volume_likelihood = create_likelihood_tables(
    price_frequency,
    volume_frequency
)

model = run_classifier(stock_data)
