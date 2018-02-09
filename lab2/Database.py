
# coding: utf-8

# In[1]:


from contextlib import contextmanager
import sqlite3
import os


# In[2]:


@contextmanager
def db(db_filename='stock.db'):
    
    conn = sqlite3.connect(os.path.join('..', 'data', db_filename), detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)

    cur = conn.cursor()
    
    yield conn, cur
    
    conn.close()


# In[3]:


def create_table_ticker():
    
    with db() as (conn, cur):
    
        cur.execute('CREATE TABLE IF NOT EXISTS ticks (stock text, date text, open real, high real, low real, close real, adjclose real, volume integer, unique (stock, date))')
        conn.commit()
    
def create_table_headlines():
    
    with db() as (conn, cur):
    
        cur.execute('CREATE TABLE IF NOT EXISTS headlines (stock text, date text, source text, content text UNIQUE ON CONFLICT IGNORE)')
        conn.commit()


# In[4]:


def add_stock_ticks(entries):
    
    with db() as (conn, cur):
    
        cur.executemany("INSERT OR IGNORE INTO ticks VALUES (?,?,?,?,?,?,?,?)", entries)
        conn.commit()
    
def add_headlines(entries):
    
    with db() as (conn, cur):
    
        cur.executemany("INSERT OR IGNORE INTO headlines VALUES (?,?,?,?)", entries)
        conn.commit()


# In[5]:


if __name__ == "__main__":
    
    create_table_ticker()
    create_table_headlines()

