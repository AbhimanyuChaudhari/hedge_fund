import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import time
import pyotp
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from kiteconnect import KiteConnect
from config.settings import settings

def update_env_token(token: str):
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    with open(env_path, 'r') as f:
        content = f.read()
    content = re.sub(r'ZERODHA_ACCESS_TOKEN=.*', f'ZERODHA_ACCESS_TOKEN={token}', content)
    with open(env_path, 'w') as f:
        f.write(content)

def get_token():
    kite = KiteConnect(api_key=settings.zerodha_api_key)
    login_url = kite.login_url()

    options = webdriver.ChromeOptions()
    # headless removed for debugging — add back once working
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )
    wait = WebDriverWait(driver, 30)

    try:
        print("Opening Zerodha login...")
        driver.get(login_url)

        # Enter user ID
        wait.until(EC.presence_of_element_located((By.ID, 'userid')))
        driver.find_element(By.ID, 'userid').send_keys(settings.zerodha_client_id)
        driver.find_element(By.ID, 'password').send_keys(settings.zerodha_password)
        driver.find_element(By.XPATH, '//button[@type="submit"]').click()

        # Enter TOTP
        totp = pyotp.TOTP(settings.zerodha_totp_secret)
        wait.until(EC.presence_of_element_located((By.XPATH, '//input[@type="number"]')))
        driver.find_element(By.XPATH, '//input[@type="number"]').send_keys(totp.now())
        driver.find_element(By.XPATH, '//button[@type="submit"]').click()

        # Capture redirect URL with request token
        time.sleep(3)
        current_url = driver.current_url
        print(f"Redirect URL: {current_url}")
        request_token = current_url.split('request_token=')[1].split('&')[0]

        # Exchange for access token
        data = kite.generate_session(request_token, api_secret=settings.zerodha_api_secret)
        access_token = data['access_token']

        update_env_token(access_token)
        print(f"Access token saved successfully.")
        print(f"Token: {access_token}")

    finally:
        driver.quit()

if __name__ == '__main__':
    get_token()