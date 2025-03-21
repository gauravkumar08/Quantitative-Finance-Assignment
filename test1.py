import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


data = yf.download("AAPL", start="2023-01-01", end="2023-12-31")

data['DPA200'] = data['Close'].rolling(window=200).mean()

data['DPA50'] = data['Close'].rolling(window=50).mean()

def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

data['Daily RSI'] = calculate_rsi(data)

data['PDEMA20'] = data['Close'].ewm(span=20, adjust=False).mean()

data = data.dropna()

print("First Few Rows of Data:")
print(data.head())

def get_number_of_stocks(investment_amount):
    if 500 <= investment_amount <= 50000:
        return 3
    elif 50001 <= investment_amount <= 1000000:
        return 4
    else:
        return 5

def calculate_max_investment_per_stock(investment_amount, num_stocks):
    return investment_amount / num_stocks

def filter_stocks(data):
    filtered_data = data[
        (data['Daily RSI'] > 60) &  
        (data['Daily RSI'] < 85)    
    ]
    return filtered_data

def sort_stocks(filtered_data):
    return filtered_data.sort_values(by=['PDEMA20', 'Daily RSI'], ascending=[False, False])

def select_top_stocks(sorted_data, num_stocks):
    return sorted_data.head(num_stocks)

def churn_portfolio(current_portfolio, new_stocks):

    current_portfolio["P&L"] = current_portfolio["Current Price"] - current_portfolio["Purchase Price"]
    

    underperforming_stocks = current_portfolio[current_portfolio["P&L"] < 0]
    if len(underperforming_stocks) > 0:
        print(f"Replacing {len(underperforming_stocks)} underperforming stocks.")
        current_portfolio = current_portfolio[current_portfolio["P&L"] >= 0]  
        current_portfolio = pd.concat([current_portfolio, new_stocks.head(len(underperforming_stocks))])  
    else:
        print("No underperforming stocks to replace.")
    
    return current_portfolio

def calculate_volatility(portfolio):
    portfolio = portfolio.copy()  
    portfolio["Daily Return"] = portfolio["Close"].pct_change()
    return portfolio["Daily Return"].std()

def plot_volatility(portfolio_volatility, nifty_volatility):
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_volatility.index, portfolio_volatility, label="Portfolio Volatility")
    plt.axhline(y=nifty_volatility, color='r', linestyle='--', label="Nifty Volatility")
    plt.title("Portfolio Volatility vs Nifty Volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.legend()
    plt.show()

def track_portfolio_value(portfolio, investment_amount):
    portfolio = portfolio.copy()  
    portfolio["Quantity"] = (investment_amount / len(portfolio)) // portfolio["Close"]
    portfolio["Current Price"] = portfolio["Close"]
    portfolio["Value"] = portfolio["Quantity"] * portfolio["Current Price"]
    return portfolio["Value"].sum()

def perform_statistical_tests(portfolio, nifty_data):
    portfolio = portfolio.copy() 
    portfolio["Daily Return"] = portfolio["Close"].pct_change()
    nifty_data["Daily Return"] = nifty_data["Close"].pct_change()
    t_stat, p_value = ttest_ind(portfolio["Daily Return"].dropna(), nifty_data["Daily Return"].dropna())
    return t_stat, p_value

if __name__ == "__main__":

    investment_amount = 100000  

    num_stocks = get_number_of_stocks(investment_amount)
    print(f"Number of stocks to purchase: {num_stocks}")

    max_investment = calculate_max_investment_per_stock(investment_amount, num_stocks)
    print(f"Max investment per stock: ₹{max_investment:.2f}")

    filtered_data = filter_stocks(data)
    if filtered_data.empty:
        print("No stocks match the filtering conditions. Please check the dataset and column names.")
    else:
        print(f"Filtered stocks: {len(filtered_data)}")

        sorted_data = sort_stocks(filtered_data)

        top_stocks = select_top_stocks(sorted_data, num_stocks)
        print("Top stocks selected:")
        print(top_stocks[["Close", "DPA200", "DPA50", "Daily RSI", "PDEMA20"]])

        current_portfolio = pd.DataFrame({
            "Stock": ["AAPL", "MSFT", "GOOGL"],
            "Purchase Price": [100, 200, 300],
            "Current Price": [110, 210, 290],
            "Quantity": [10, 5, 3]
        })
        new_portfolio = churn_portfolio(current_portfolio, top_stocks)
        print("New portfolio after churning:")
        print(new_portfolio[["Stock", "Purchase Price", "Current Price", "Quantity", "P&L"]])

        portfolio_volatility = calculate_volatility(top_stocks)
        print(f"Portfolio volatility: {portfolio_volatility:.4f}")

        nifty_data = yf.download("^NSEI", period="1mo")  
        nifty_volatility = nifty_data["Close"].pct_change().std().item()  
        plot_volatility(top_stocks["Close"].pct_change(), nifty_volatility)

        portfolio_value = track_portfolio_value(top_stocks, investment_amount)
        print(f"Final portfolio value: ₹{portfolio_value:.2f}")

        t_stat, p_value = perform_statistical_tests(top_stocks, nifty_data)
        print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")

    data['Close'].plot(title="AAPL Closing Prices (2023)", figsize=(10, 6))
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.show()

    data['Daily RSI'].plot(title="AAPL Daily RSI (2023)", figsize=(10, 6))
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.show()

    data[['DPA200', 'DPA50']].plot(title="AAPL Moving Averages (2023)", figsize=(10, 6))
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.show()