import requests
from dateutil import parser
import time
import io
import pandas as pd
from bs4 import BeautifulSoup

BASE_URL = "https://finance.yahoo.com/quote/{ticker}/history"
CSV_URL = "https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={start}&period2={end}&interval=1d&events=history&crumb={crumb}"

MAX_ATTEMPTS = 3

def get_stock_prices(tickers, startDate, endDate):
    """
    Returns a Pandas Panel containing the daily HLOC and adjusted close for TICKER from STARTDATE to ENDDATE.
    The items of the panel are:
    * "Open"
    * "High"
    * "Low"
    * "Close"
    * "Adj Close"
    * "Volume"
    """
    try:
        start_time = parser.parse(startDate)
        end_time = parser.parse(endDate)
    except:
        print("Error: Could not parse date given. Please input date as a string, e.g., '2017-03-30', or '2017/3/30', etc.")
        return
        
    start_epoch = int(time.mktime(start_time.timetuple()))
    end_epoch = int(time.mktime(end_time.timetuple()))
    
    if start_epoch > end_epoch:
        print("Error: Start date must be before end date")
        return
    
    tickers = [ticker.upper() for ticker in tickers]
    
    attempts = 0
    success = False
    
    start = time.time()
    while (not success):
        try:
            panel = _get_panel(start_epoch, end_epoch, tickers)
            success = True
        except:
            attempts += 1
            if attempts >= MAX_ATTEMPTS:
                print("Error: Could not access Yahoo Finance data after %d attempts." % MAX_ATTEMPTS)
                break
            continue
        
    end = time.time()
    
    if success:
        print("Succeeded in %.2fs after %d attempts" % (end - start, attempts + 1))
        return panel
    else:
        print("Failed to load after %d attempts" % attempts + 1)

def _get_panel(start_epoch, end_epoch, tickers):
    with requests.Session() as s:
        r = s.get(BASE_URL.format(ticker=tickers[0]))
        soup = BeautifulSoup(r.content, 'html.parser')
        html = soup.prettify()
        crumb_start = html.find("CrumbStore") + 22
        crumb_end = crumb_start + html[crumb_start: crumb_start + 30].find('\"')
        crumb = soup.prettify()[crumb_start:crumb_end]
        
        df_dict = {column: pd.DataFrame() for column in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']}
        
        for ticker in tickers:
            csv = s.get(CSV_URL.format(ticker=ticker, start=start_epoch, end=end_epoch, crumb=crumb)).content
            df = pd.read_csv(io.StringIO(csv.decode('utf-8')), index_col=0)
            
            for column in df_dict.keys():
                column_df = df_dict[column]
                column_df[ticker] = df[column]
                df_dict[column] = column_df
                
        full_panel = pd.Panel(df_dict)
        
    return full_panel