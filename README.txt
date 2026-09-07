Student Name: Saravenus A Khon
Initial Date: 09/01/2026
Project - CTA06 - Fun with Naive Bayes Classifier

I chose this project idea to apply the Naive Bayes classifier to a real-world scenario that was easy to understand and measure.
I wanted to learn more about the stock market, so I chose Carnival Corporation stock data because the outcome could be framed as a
simple classification problem: whether the stock price would be higher after five trading days.
I used two binary features: whether the current stock price was above its 20-day average and whether the trading volume was above its 20-day average.
This worked well with the Bernoulli Naive Bayes model because the features could be represented as 0 or 1 values.
The project also meets the assignment requirements by creating frequency tables, converting counts into likelihood probabilities,
applying Laplace correction to prevent zero-probability errors, and calculating the posterior probability for both classes before showing the final prediction.
Below are instructions of how to run this script.


Instructions for basic testing

1: Make sure Python is Installed

2: verify the following libraries are also installed:
    -Pandas
    -Scikit-learn

3: Make a folder called : Data

4: Put the CCL_HistoricaData_5years.csv file inside the folder name, "data" in the same project folder as the Py.script.

5: Open the naive_bayes_stock_classifier.py in an IDE

6: Run the script

7: The Script will do the following:
    - Load and prepare the Carnival Corporation historical stock data
    -make the frequency tables
    -make the likelihood tables
    -Apply Laplace correction to stop zero-probability errors.
    -Train a Bernoulli Naive Bayes classifier
    - Calculate the posterior probability y for each class
    -Show the final stock classification prediction


8: A successful test will run without errors and display:
    - Frequency Tables
    - Likelihood Tables
    - Posterior Probabilities
    - Final Predictions

