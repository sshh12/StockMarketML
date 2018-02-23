
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


def create_tables():
    
    with db() as (conn, cur):
    
        cur.execute('CREATE TABLE IF NOT EXISTS ticks (stock text, date text, open real, high real, low real, close real, adjclose real, volume integer, unique (stock, date))')
        conn.commit()
        
        cur.execute('CREATE TABLE IF NOT EXISTS headlines (stock text, date text, source text, content text UNIQUE ON CONFLICT IGNORE, rawcontent text UNIQUE ON CONFLICT IGNORE, sentimentlabel integer)')
        conn.commit()      
        
        cur.execute('CREATE TABLE IF NOT EXISTS dictionary (word text, stock text, replacement text, unique (word, stock, replacement))')
        conn.commit()


# In[4]:


def add_stock_ticks(entries):
    
    with db() as (conn, cur):
    
        cur.executemany("INSERT OR IGNORE INTO ticks VALUES (?,?,?,?,?,?,?,?)", entries)
        conn.commit()
    
def add_headlines(entries):
    
    with db() as (conn, cur):
    
        cur.executemany("INSERT OR IGNORE INTO headlines VALUES (?,?,?,?,?,?)", entries)
        conn.commit()
        
def clean_ticks():
    
    with db() as (conn, cur):
    
        cur.execute("DELETE FROM ticks WHERE adjclose='null'")
        conn.commit()


# In[5]:


def db_replace_all(query, replacement, stock='', commit=False):
    """
    Do a `replace all` on headlines in database
    
    Used to fix parsing errors of already parsed headlines
    """
    with db() as (conn, cur):
        
        cur.execute("SELECT stock, date, source, content FROM headlines WHERE stock=? AND content LIKE ?", [stock, query])
        results = cur.fetchall()
        
        for (stock, date, source, content) in results:
            
            new_content = content.replace(query.replace('%', ''), replacement)
            
            print(content)
            print(new_content)
            
            if commit:
                
                cur.execute("UPDATE headlines SET content=? WHERE stock=? AND content=?", [new_content, stock, content])
                conn.commit()
        


# In[6]:


if __name__ == "__main__":
    
    create_tables()


# In[7]:


if __name__ == "__main__":
    
    ## Populate Database of Words ##
    
    hardcoded_dict = [
        ## Special Generic Tokens ##
        ["**STATISTIC**"], 
        ### Company ###
        ["**COMPANY**"], 
        ["**COMPANY**owned"],
        ["**COMPANY**s"],
        ["**COMPANY**es"],
        ["**COMPANY**stock"],
        ["ex**COMPANY**"],
        ["**COMPANY**made"],
        ["madeby**COMPANY**"],
        ["**COMPANY**com"],
        ["r**COMPANY**stock"],
        ["non**COMPANY**"],
        ["**COMPANY**insider"],
        ["nasdaq**COMPANY**"],
        ["**COMPANY**cloud"],
        ["un**COMPANY**like"],
        ["**COMPANY**like"],
        ["**COMPANY**only"],
        ["**COMPANY**certified"],
        ["anti**COMPANY**"],
        ["**COMPANY**esque"],
        ### Product ###
        ["**PRODUCT**phones"],
        ["**PRODUCT**com"],
        ["**PRODUCT**"],
        ["**PRODUCT**powered"],
        ["team**PRODUCT**"],
        ["**PRODUCT**like"],
        ["**PRODUCT**tablet"],
        ["**PRODUCT**authority"],
        ["ultra**PRODUCT**"],
        ["**PRODUCT**insiders"],
        ["**PRODUCT**enabled"],
        ["r**PRODUCT**s"],
        ["**PRODUCT**based"],
        ["**PRODUCT**only"],
        ["anti**PRODUCT**"],
        ["**PRODUCT**tm"],
        ["**PRODUCT**s"],
        ["non**PRODUCT**"],
        ### Member ###
        ["**MEMBER**"],
        ["**MEMBERs**"],
        ## Common (non-glove) Tokens ##
        ["singlecore"],
        ["nowassistant"],
        ["assistantenabled"],
        ["deeplearning"],
        ["wannacry"],
        ["wannacrypt"],
        ["qualcomms"],
        ["smartglasses"],
        ["selfdriving"],
        ["pichai"],
        ["zuckerberg"],
        ["geekwire"],
        ["uscanada"],
        ["outofstock"],
        ["outofsale"],
        ["demonetizing"],
        ["hydrogenpowered"],
        ["homebutton"],
        ["electriccar"],
        ["xamarin"],
        ["wellcalibrated"],
        ["antitrump"],
        ["multigpu"],
        ["voicerecognition"],
        ["firstgen"],
        ["secondgeneration"],
        ["thirdgeneration"],
        ["investbuy"],
        ["nearzero"],
        ["techsavvy"],
        ["steamvr"],
        ["mostlycomplete"],
        ["anticlimate"],
        ["taxbonuses"],
        ["steamos"],
        ["specialedition"],
        ["testdriving"],
        ["oneplus"],
        ["airpods"],
        ["lyft"],
        ["stockbased"],
        ["multiadapter"],
        ["tvstick"],
        ["militarygrade"],
        ["kotlin"],
        ["stockindex"],
        ["marketmoving"],
        ["belkins"],
        ["subscriptionbased"],
        ["airpod"],
        ["chiprelated"],
        ["cheapbudget"],
        ["scaledup"],
        ["wirecutter"],
        ["wristworn"],
        ["biggerscreen"],
        ["employeeranking"],
        ["stockportfolio"],
        ["firetv"],
        ["floppydisk"],
        ["zerocommission"],
        ["zenwatch"],
        ["nsagov"],
        ["fbis"],
        ["smartpost"],
        ["conflictfree"],
        ["closedsource"],
        ["communityrun"],
        ["voiceactivated"],
        ["voiceordered"],
        ["safetynet"],
        ["gsync"],
        ["chromecast"],
        ["jpmorgans"],
        ["septupled"],
        ["liquidresistant"],
        ["brexit"],
        ["meltdownspectre"],
        ["multiwaypoint"],
        ["carplay"],
        ["hotword"],
        ["nearstock"],
        ["mindblank"],
        ["thermalpower"],
        ["datascience"],
        ["spotifys"],
        ["bigheavy"],
        ["stockpicks"],
        ["mediastore"],
        ["forceclosed"],
        ["trudeaus"],
        ["upgradekits"],
        ["autoorientation"],
        ["ryzen"],
        ["computershared"],
        ["hypetrain"],
        ["adaptercharger"],
        ["bigbasket"],
        ["muslimban"],
        ["samsungbuilt"],
        ["unesthetic"],
        ["outlookcom"],
        ["postsnowden"],
        ["hashrate"],
        ["yoloed"],
        ["halfsalary"],
        ["neweggcom"],
        ["hyperlapse"],
        ["snapchats"],
        ["browserhijacking"],
        ["iconpack"],
        ["vpnbased"],
        ["onecore"],
        ["russianlinked"],
        ["playerunknowns"],
        ["russianbought"],
        ["instacart"],
        ["intelbacked"],
        ["pointzero"],
        ["snapchatlike"],
        ["housesharing"],
        ["waymo"],
        ["bugglitch"],
        ["xiaomi"],
        ["driverlesscar"],
        ["selfdrivingcar"],
        ["axpowered"],
        ["intelliglass"],
        ["aiinfused"],
        ["hathawaylike"],
        ["airdelivery"],
        ["trumpian"],
        ["higheragain"],
        ["batterieskeep"],
        ["topreferrer"],
        ["litecoin"],
        ["gearvr"],
        ["lightsail"],
        ["autovoice"],
        ["dnapowered"],
        ["profitsurging"],
        ["ballooninternet"],
        ["amazonvisa"],
        ["antiencryption"],
        ["undisruptive"],
        ["lumentum"],
        ["renderingtext"],
        ["expectationcrushing"],
        ["microbezel"],
        ["bezelless"],
        ["cryptocurrencies"],
        ["softcard"],
        ["millionairemaker"],
        ["debtpowered"],
        ["nonxl"],
        ["panoramafunction"],
        ["applesized"],
        ["quartercrushing"],
        ["internetofthings"],
        ["marginexpansion"],
        ["linkedins"],
        ["phoneshaped"],
        ["chipgate"],
        ## Specialized Tokens ##
        ### MSFT ###
        ["onedrive", "MSFT", "**PRODUCT**"],
        ["bing", "MSFT", "**PRODUCT**"],
        ["xbox one x", "MSFT", "**PRODUCT**"],
        ["satya nadella", "MSFT", "**MEMBER**"],
        ["microsoft", "MSFT", "**COMPANY**"],
        ["outlook", "MSFT", "**PRODUCT**"],
        ["xbox one", "MSFT", "**PRODUCT**"],
        ["windows", "MSFT", "**PRODUCT**"],
        ["xbox", "MSFT", "**PRODUCT**"],
        ### AMD ###
        ["radeon rxvega", "AMD", "**PRODUCT**"],
        ["rxvega", "AMD", "**PRODUCT**"],
        ["advanced micro devices", "AMD", "**COMPANY**"],
        ["radeon vega frontier edition", "AMD", "**PRODUCT**"],
        ["lisa su", "AMD", "**MEMBER**"],
        ["radeon", "AMD", "**PRODUCT**"],
        ["zen", "AMD", "**PRODUCT**"],
        ["amd", "AMD", "**COMPANY**"],
        ["vega fe", "AMD", "**PRODUCT**"],
        ["ryzen", "AMD", "**PRODUCT**"],
        ### AMZN ###
        ["echo", "AMZN", "**PRODUCT**"],
        ["alexa", "AMZN", "**PRODUCT**"],
        ["prime video", "AMZN", "**PRODUCT**"],
        ["firephone", "AMZN", "**PRODUCT**"],
        ["amazon", "AMZN", "**COMPANY**"],
        ["jeff bezos", "AMZN", "**MEMBER**"],
        ["amazondot", "AMZN", "**PRODUCT**"],
        ["firetv", "AMZN", "**PRODUCT**"],
        ["jeffbezos", "AMZN", "**MEMBER**"],
        ["amazonfire", "AMZN", "**PRODUCT**"],
        ["amazonfresh", "AMZN", "**PRODUCT**"],
        ["dot", "AMZN", "**PRODUCT**"],
        ["amazonvisa", "AMZN", "**PRODUCT**"],
        ["prime", "AMZN", "**PRODUCT**"],
        [" dash", "AMZN", " **PRODUCT**"],
        ["smileamazoncom", "AMZN", "**PRODUCT**"],
        ### AAPL ###
        ["lightningpin", "AAPL", "**PRODUCT**"],
        ["siri", "AAPL", "**PRODUCT**"],
        ["iphone", "AAPL", "**PRODUCT**"],
        ["iphone x", "AAPL", "**PRODUCT**"],
        ["applepark", "AAPL", "**PRODUCT**"],
        ["iphonex", "AAPL", "**PRODUCT**"],
        ["airpods", "AAPL", "**PRODUCT**"],
        ["macbook", "AAPL", "**PRODUCT**"],
        ["appletv", "AAPL", "**PRODUCT**"],
        ["applepay", "AAPL", "**PRODUCT**"],
        ["apple", "AAPL", "**COMPANY**"],
        ["imessage", "AAPL", "**PRODUCT**"],
        ["icloud", "AAPL", "**PRODUCT**"],
        ["applewatch", "AAPL", "**PRODUCT**"],
        ["facetime", "AAPL", "**PRODUCT**"],
        ["animoji", "AAPL", "**PRODUCT**"],
        ["faceid", "AAPL", "**PRODUCT**"],
        ["face id", "AAPL", "**PRODUCT**"],
        ["imac", "AAPL", "**PRODUCT**"],
        ["ios", "AAPL", "**PRODUCT**"],
        ["d touch", "AAPL", "**PRODUCT**"],
        ["ipad", "AAPL", "**PRODUCT**"],
        ["touchid", "AAPL", "**PRODUCT**"],
        ["tim cook", "AAPL", "**MEMBER**"],
        ["applemusic", "AAPL", "**PRODUCT**"],
        ### GOOG ###
        ["nexusx", "GOOG", "**PRODUCT**"],
        ["pixel xl", "GOOG", "**PRODUCT**"],
        ["pixelxl", "GOOG", "**PRODUCT**"],
        ["pixel", "GOOG", "**PRODUCT**"],
        ["chrome", "GOOG", "**PRODUCT**"],
        ["googlephotos", "GOOG", "**PRODUCT**"],
        ["chromecast", "GOOG", "**PRODUCT**"],
        ["chromebook", "GOOG", "**PRODUCT**"],
        ["googleplay", "GOOG", "**PRODUCT**"],
        ["android", "GOOG", "**PRODUCT**"],
        ["nexusp", "GOOG", "**PRODUCT**"],
        ["nexus", "GOOG", "**PRODUCT**"],
        ["maps", "GOOG", "**PRODUCT**"],
        [" allo ", "GOOG", " **PRODUCT** "],
        ["youtube", "GOOG", "**PRODUCT**"],
        ["googletranslate", "GOOG", "**PRODUCT**"],
        ["alphabet", "GOOG", "**COMPANY**"],
        ["androidpay", "GOOG", "**PRODUCT**"],
        ["googlehome", "GOOG", "**PRODUCT**"],
        ["play store", "GOOG", "**PRODUCT**"],
        ["googlemusic", "GOOG", "**PRODUCT**"],
        ["sundar pichai", "GOOG", "**MEMBER**"],
        ["gboard", "GOOG", "**PRODUCT**"],
        ["alphago", "GOOG", "**PRODUCT**"],
        ["sundarpichai", "GOOG", "**MEMBER**"],
        ["materialdesign", "GOOG", "**PRODUCT**"],
        ["google", "GOOG", "**COMPANY**"]
    ]
    
    for item in hardcoded_dict:
        
        if len(item) == 1:
            item.extend(['none', item[0]])
    
    with db() as (conn, cur):
    
        cur.executemany("INSERT OR IGNORE INTO dictionary VALUES (?,?,?)", hardcoded_dict)
        conn.commit()
    

