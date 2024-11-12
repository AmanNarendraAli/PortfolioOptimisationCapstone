from splinter import Browser
from selenium import webdriver
import time, os
import pandas as pd
download_dir = "/tmp"
bse_link = "'https://mock.bseindia.com/corporates/List_Scrips.html'"

# change download directory if required
prefs = {"download.default_directory": download_dir};

options = webdriver.ChromeOptions()
options.add_experimental_option("prefs", prefs)

# intiate browser 
browser = Browser('chrome', options=options, headless=True)

# visit link
browser.visit(bse_link)

# fill out form fields
browser.find_by_id('ddlsegment').select("Equity")
browser.find_by_id('ddlstatus').select("Active")

# hit submit button
browser.find_by_id('btnSubmit').click()

# let the table load 
browser.is_element_present_by_text("Issuer Name")
time.sleep(5)

# download
browser.find_by_id('lnkDownload').click()

df_bse = pd.read_csv(os.path.join(download_dir, "Equity.csv"))