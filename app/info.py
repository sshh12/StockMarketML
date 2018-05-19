from contextlib import contextmanager
import requests
import sqlite3
import json
import math
import os
import re


DB_URL = os.path.join('..', 'data', 'stock.db')

@contextmanager
def db(db_filename=DB_URL):

    conn = sqlite3.connect(db_filename, detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)

    cur = conn.cursor()

    yield conn, cur

    conn.close()

class YahooStock(object):

    re_extract_json = re.compile("root.App.main = {[\s\S]+?;\s}\(this\)")

    def __init__(self, symbol):

        self.symbol = symbol

    def download(self):

        raw_page = requests.get("https://finance.yahoo.com/quote/{0}?p={0}".format(self.symbol)).text

        raw_data = self.re_extract_json.search(raw_page).group(0)
        raw_data = raw_data.replace("root.App.main = ", "").replace("}(this)", "").replace(";", "")
        self.data = json.loads(raw_data)
        self.quote_sum = self.data["context"]["dispatcher"]["stores"]["QuoteSummaryStore"]

    @property
    def summary(self):
        return self.quote_sum["summaryProfile"]

    @property
    def quote_data(self):
        return self.quote_sum["quoteType"]

    @property
    def finanical_data(self):
        return self.quote_sum["financialData"]

    @property
    def details(self):
        return self.quote_sum["summaryDetail"]

    @property
    def price(self):
        return self.quote_sum["price"]

def get_tracked_stocks():

    with db() as (conn, cur):

        cur.execute("SELECT DISTINCT stock FROM ticks ORDER BY stock ASC")
        return [val[0] for val in cur.fetchall()]

def get_historical_closes(symbol, n=35):

    with db() as (conn, cur):

        cur.execute("SELECT adjclose FROM ticks WHERE stock=? ORDER BY date DESC LIMIT ?", [symbol, n])
        return list(reversed([val[0] for val in cur.fetchall()]))

def main():

    stocks = get_tracked_stocks()

    stock_data = []

    for stock in stocks:

        yahoo = YahooStock(stock)
        yahoo.download()

        historical_data = get_historical_closes(stock)
        chart_data = [close / historical_data[0] for close in historical_data]

        stock_data.append({
            "symbol": stock,
            "shortname": yahoo.quote_data["shortName"],
            "fullname": yahoo.quote_data["longName"],
            "sector": yahoo.summary["sector"],
            "price": yahoo.price["regularMarketPrice"]["fmt"],
            "change": yahoo.price["regularMarketChange"]["fmt"],
            "changepercent": yahoo.price["regularMarketChangePercent"]["fmt"],
            "color": ("green" if yahoo.price["regularMarketChange"]["raw"] > 0 else "red"),
            "barwidth": math.log(abs(yahoo.price["regularMarketChangePercent"]["raw"]) * 100 + 1) * 30,
            "chartdata": ",".join(str(v) for v in chart_data)
        })

    data = {
        "stocks": stock_data
    }

    return data
