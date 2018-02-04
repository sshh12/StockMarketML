
# coding: utf-8

# In[1]:


import sqlite3
import os


# In[2]:


def connect(db_filename='stock.db'):
    
    conn = sqlite3.connect(os.path.join('..', 'data', db_filename), detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)

    cur = conn.cursor()
    
    return conn, cur


# In[3]:


def create_table_ticker():
    
    conn, cur = connect()
    
    cur.execute('CREATE TABLE ticks (stock text, date text, open real, high real, low real, close real, adjclose real, volume integer)')
    conn.commit()
    
    conn.close()


# In[4]:


def add_stock_ticks(entries):
    
    conn, cur = connect()
    
    cur.executemany("INSERT INTO ticks VALUES (?,?,?,?,?,?,?,?)", entries)
    conn.commit()
    
    conn.close()

