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
    env_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'
    )
    with open(env_path, 'r') as f:
        content = f.read()
    if 'ZERODHA_ACCESS_TOKEN=' in content:
        content = re.sub(
            r'ZERODHA_ACCESS_TOKEN=.*',
            f'ZERODHA_ACCESS_TOKEN={token}',
            content
        )
    else:
        content += f'\nZERODHA_ACCESS_TOKEN={token}'
    with open(env_path, 'w') as f:
        f.write(content)


def click_visible_button(driver):
    """Click the first visible button on the page."""
    try:
        # Try submit button first
        btn = driver.find_element(By.CSS_SELECTOR, 'button[type="submit"]')
        driver.execute_script("arguments[0].click();", btn)
        return
    except Exception:
        pass
    try:
        # Try any visible button via JavaScript
        driver.execute_script("""
            var btns = document.querySelectorAll('button');
            for (var i = 0; i < btns.length; i++) {
                if (btns[i].offsetParent !== null && btns[i].offsetWidth > 0) {
                    btns[i].click();
                    break;
                }
            }
        """)
    except Exception as e:
        print(f'[WARN] Button click failed: {e}')


def get_totp_field(driver, wait):
    """Find the TOTP input field — tries multiple selectors."""
    selectors = [
        'input[type="number"]',
        'input[placeholder*="OTP"]',
        'input[placeholder*="TOTP"]',
        'input[placeholder*="code"]',
        'input[placeholder*="2FA"]',
        '#userid',
        'input.otp',
    ]
    for sel in selectors:
        try:
            field = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, sel))
            )
            if field.is_displayed():
                return field
        except Exception:
            continue
    # Fallback — any input visible on page
    inputs = driver.find_elements(By.TAG_NAME, 'input')
    for inp in inputs:
        if inp.is_displayed() and inp.get_attribute('type') in ('number', 'text', 'tel'):
            return inp
    raise RuntimeError('Could not find TOTP input field')


def get_token():
    kite      = KiteConnect(api_key=settings.zerodha_api_key)
    login_url = kite.login_url()

    options = webdriver.ChromeOptions()
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1280,800')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option('excludeSwitches', ['enable-automation'])

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )
    wait = WebDriverWait(driver, 30)

    try:
        print('Opening Zerodha login...')
        driver.get(login_url)
        time.sleep(2)

        # ── Step 1: Enter credentials ──────────────────────────────
        wait.until(EC.presence_of_element_located((By.ID, 'userid')))
        driver.find_element(By.ID, 'userid').clear()
        driver.find_element(By.ID, 'userid').send_keys(settings.zerodha_client_id)
        time.sleep(0.5)
        driver.find_element(By.ID, 'password').clear()
        driver.find_element(By.ID, 'password').send_keys(settings.zerodha_password)
        time.sleep(0.5)
        click_visible_button(driver)
        time.sleep(3)
        print(f'After login: {driver.current_url}')

        # ── Step 2: Enter TOTP ─────────────────────────────────────
        totp      = pyotp.TOTP(settings.zerodha_totp_secret)
        totp_code = totp.now()
        time.sleep(1)

        totp_field = get_totp_field(driver, wait)
        totp_field.clear()
        totp_field.send_keys(totp_code)
        time.sleep(0.5)
        click_visible_button(driver)
        time.sleep(4)

        current_url = driver.current_url
        print(f'Final URL: {current_url}')

        if 'request_token=' not in current_url:
            print(f'[ERROR] No request_token in URL: {current_url}')
            print(f'Page source snippet: {driver.page_source[:500]}')
            return

        request_token = current_url.split('request_token=')[1].split('&')[0]
        data          = kite.generate_session(
            request_token,
            api_secret=settings.zerodha_api_secret
        )
        access_token = data['access_token']

        update_env_token(access_token)
        print(f'Access token saved successfully.')
        print(f'Token: {access_token}')

    except Exception as e:
        print(f'[ERROR] {e}')
        try:
            print(f'Current URL: {driver.current_url}')
        except Exception:
            pass

    finally:
        driver.quit()


if __name__ == '__main__':
    get_token()