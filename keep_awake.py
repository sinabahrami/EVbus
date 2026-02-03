import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# Define options for headless mode
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# Specify the URL of your Streamlit app
app_url = "https://transit-electrification.streamlit.app/"

# Set up the service and driver (GitHub Actions environment will handle the binary)
try:
    service = Service()
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    print(f"Navigating to {app_url}")
    driver.get(app_url)
    
    # Keep the browser open for a short period to register the session
    time.sleep(60) 
    print("Page loaded and active for 60 seconds.")
    
except Exception as e:
    print(f"An error occurred: {e}")
    
finally:
    if driver:
        driver.quit()
        print("Browser closed.")
