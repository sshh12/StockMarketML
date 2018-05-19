from contextlib import contextmanager
import requests
import sqlite3
import json
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

    @property
    def summary(self):
        return self.data["context"]["dispatcher"]["stores"]["QuoteSummaryStore"]["summaryProfile"]

    @property
    def quote_data(self):
        return self.data["context"]["dispatcher"]["stores"]["QuoteSummaryStore"]["quoteType"]

    @property
    def finanical_data(self):
        return self.data["context"]["dispatcher"]["stores"]["QuoteSummaryStore"]["financialData"]

    @property
    def details(self):
        return self.data["context"]["dispatcher"]["stores"]["QuoteSummaryStore"]["summaryDetail"]

    @property
    def price(self):
        return self.data["context"]["dispatcher"]["stores"]["QuoteSummaryStore"]["price"]

def get_tracked_stocks():

    with db() as (conn, cur):

        cur.execute("SELECT DISTINCT stock FROM ticks ORDER BY stock ASC")
        return [val[0] for val in cur.fetchall()]

def main():

    stocks = get_tracked_stocks()

    stock_data = []

    for stock in stocks:

        yahoo = YahooStock(stock)
        yahoo.download()

        stock_data.append({
            "symbol": stock,
            "shortname": yahoo.quote_data["shortName"],
            "fullname": yahoo.quote_data["longName"],
            "sector": yahoo.summary["sector"],
            "price": yahoo.price["regularMarketPrice"]["fmt"],
            "change": yahoo.price["regularMarketChange"]["fmt"],
            "changepercent": yahoo.price["regularMarketChangePercent"]["fmt"]
        })

    data = {
        "stocks": stock_data
    }

    return data
