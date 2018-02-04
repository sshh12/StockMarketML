
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
    
def create_table_headlines():
    
    conn, cur = connect()
    
    cur.execute('CREATE TABLE headlines (stock text, date text, source text, content text UNIQUE ON CONFLICT IGNORE)')
    conn.commit()
    
    conn.close()


# In[4]:


def add_stock_ticks(entries):
    
    conn, cur = connect()
    
    cur.executemany("INSERT INTO ticks VALUES (?,?,?,?,?,?,?,?)", entries)
    conn.commit()
    
    conn.close()
    
def add_headlines(entries):
    
    conn, cur = connect()
    
    cur.executemany("INSERT INTO headlines VALUES (?,?,?,?)", entries)
    conn.commit()
    
    conn.close()


# In[5]:


if __name__ == "__main__":
    
    create_table_ticker()
    create_table_headlines()

