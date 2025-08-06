#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Cell 1: Import Libraries and Install Requirements
# Install required packages

get_ipython().system('pip install azure-storage-blob')
get_ipython().system('pip install crawlbase')

# Import all the tools we need
import requests
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import json
import time
from crawlbase import CrawlingAPI

# HIP2 SCRIPT START TIMER (after imports)
script_start_time = datetime.now()
print(f"🚀 HIP Script Started: {script_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

# Check if imports worked
print("All libraries imported successfully!")
print(f"Current time: {datetime.now()}")


# In[2]:


# HIP2 Target hotels
hotels = [
    {"name": "Artezen Hotel", "id": "21542436"},
    {"name": "French Quarters NYC", "id": "36859797"},  # Hotel Cherman
    {"name": "CitizenM New York Times Square", "id": "8356903"},
    {"name": "Hotel Chelsea", "id": "1551375"},
    {"name": "The Wallace", "id": "896126"},
    {"name": "Hotel Riu Plaza Manhattan Times Square", "id": "60677495"}
]

# Display hotels
for hotel in hotels:
    print(f"{hotel['name']} (ID: {hotel['id']})")


# In[3]:


# Add this after Cell 2 to verify hotels
print(f"\nVerifying {len(hotels)} hotels:")
for i, hotel in enumerate(hotels):
    print(f"{i+1}. {hotel['name']} (ID: {hotel['id']})")
print(f"\nTotal unique hotel IDs: {len(set(h['id'] for h in hotels))}")


# In[4]:


# Cell 3: Create config file (only if it doesn't exist)
import os

if os.path.exists('hip_config.txt'):
    print("✓ Config file already exists! Not overwriting.")
    print("  To add/change your token, edit 'hip_config.txt' directly.")
else:
    config_content = """# Configuration for HIP
CRAWLBASE_TOKEN=your_token_here
"""
    with open('hip_config.txt', 'w') as f:
        f.write(config_content)
    
    print("✅ Config file created!")
    print("  Now edit 'hip_config.txt' and add your Crawlbase JavaScript token")


# In[5]:


# Function to read your token safely
def get_crawlbase_token():
    """Read token from config file"""
    try:
        with open('hip_config.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('CRAWLBASE_TOKEN='):
                    token = line.split('=')[1].strip()
                    return token
        print("Token not found in config file!")
        return None
    except FileNotFoundError:
        print("Config file not found! Run cell 3 first.")
        return None

# Test loading token
token = get_crawlbase_token()
if token and token != 'your_token_here':
    print("✓ Token loaded successfully!")
    print(f"Token starts with: {token[:10]}...")
else:
    print("✗ Please add your token to hip_config.txt")


# In[6]:


# Let's build a URL for one test case
hotel = hotels[0]  # Pendry Manhattan West
check_in = datetime.now() + timedelta(days=7)  # 7 days from now
check_out = check_in + timedelta(days=1)  # Next day

# Format dates
check_in_str = check_in.strftime('%Y-%m-%d')
check_out_str = check_out.strftime('%Y-%m-%d')

# Build URL
base_url = f"https://www.expedia.com/New-York-Hotels-{hotel['name'].replace(' ', '-')}"
full_url = f"{base_url}.h{hotel['id']}.Hotel-Information?chkin={check_in_str}&chkout={check_out_str}"

print(f"Hotel: {hotel['name']}")
print(f"Check-in: {check_in_str}")
print(f"Check-out: {check_out_str}")
print(f"\nURL: {full_url[:100]}...")  # Show first 100 characters


# In[7]:


# Enhanced Cell 6: Crawlbase with Retry Logic and Monitoring
from crawlbase import CrawlingAPI
import time
import random
from collections import deque
from datetime import datetime

# Global variables for monitoring (AI optimization requirements)
request_history = deque(maxlen=100)  # Track last 100 requests
consecutive_requests = 0
session_start_time = datetime.now()

def log_request_result(success, retry_count=0):
    """Log request results for AI optimization monitoring"""
    global request_history, consecutive_requests
    
    timestamp = datetime.now()
    request_history.append({
        'timestamp': timestamp,
        'success': success,
        'retry_count': retry_count
    })
    
    if success:
        consecutive_requests += 1
    
    # Print monitoring info every 10 requests
    if len(request_history) % 10 == 0:
        recent_success_rate = sum(1 for r in list(request_history)[-10:] if r['success']) / 10
        print(f"📊 Last 10 requests: {recent_success_rate*100:.0f}% success rate")
        
        if consecutive_requests >= 50:
            print(f"✅ {consecutive_requests} consecutive requests completed - AI optimization active!")

def fetch_with_crawlbase(url, token, max_retries=3, base_delay=2):
    """
    Fetch URL using Crawlbase with retry logic and monitoring
    
    Args:
        url: Target URL to scrape
        token: Crawlbase token
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Base delay for exponential backoff in seconds (default: 2)
    
    Returns:
        HTML string if successful, None if all retries failed
    """
    
    print(f"🔍 Fetching: {url[:80]}...")
    
    # Initialize API
    try:
        api = CrawlingAPI({'token': token, 'timeout': 60})  # 1 minute timeout
    except Exception as e:
        print(f"❌ Failed to initialize Crawlbase API: {e}")
        log_request_result(False)
        return None
    
    # Options for JavaScript rendering
    options = {
        'ajax_wait': True,
        'page_wait': 2000,  # Wait 2 seconds (faster)
        'format': 'html'  # Ensure we get HTML format
    }
    
    # Retry loop
    for attempt in range(max_retries + 1):  # +1 for initial attempt
        try:
            if attempt > 0:
                # Exponential backoff with jitter
                delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
                print(f"⏳ Retry {attempt}/{max_retries} after {delay:.1f}s delay...")
                time.sleep(delay)
            
            # Make the request
            response = api.get(url, options)
            
            # Check response status
            status_code = response.get('status_code', 0)
            
            if status_code == 200:
                # Success!
                html = response.get('body', '')
                
                # Handle bytes response (convert to string if needed)
                if isinstance(html, bytes):
                    html = html.decode('utf-8', errors='ignore')
                
                if html and len(html) > 100:  # Basic validation
                    # Log additional status info
                    headers = response.get('headers', {})
                    original_status = headers.get('original_status', 'N/A')
                    pc_status = headers.get('pc_status', 'N/A')
                    
                    print(f"✅ Success! HTML: {len(html):,} chars")
                    print(f"   Original status: {original_status} | Crawlbase status: {pc_status}")
                    
                    if attempt > 0:
                        print(f"   ↻ Succeeded on retry {attempt}")
                    
                    log_request_result(True, attempt)
                    return html
                else:
                    print(f"⚠️  Received empty/invalid HTML (length: {len(html) if html else 0})")
                    if attempt == max_retries:
                        log_request_result(False, attempt)
                        return None
                    continue
            
            else:
                # Failed status code
                error_msg = str(response.get('body', 'Unknown error'))[:100]
                print(f"❌ Status {status_code}: {error_msg}")
                
                if attempt == max_retries:
                    log_request_result(False, attempt)
                    return None
                continue
                
        except Exception as e:
            print(f"❌ Request error: {type(e).__name__}: {str(e)[:100]}")
            
            if attempt == max_retries:
                log_request_result(False, attempt)
                return None
            continue
    
    # If we get here, all retries failed
    print(f"💥 All {max_retries + 1} attempts failed")
    log_request_result(False, max_retries)
    return None

def print_session_stats():
    """Print comprehensive session statistics"""
    global request_history, consecutive_requests, session_start_time
    
    if not request_history:
        print("📊 No requests made yet")
        return
    
    total_requests = len(request_history)
    successful_requests = sum(1 for r in request_history if r['success'])
    success_rate = successful_requests / total_requests * 100
    
    # Calculate retry statistics
    total_retries = sum(r.get('retry_count', 0) for r in request_history)
    avg_retries = total_retries / total_requests if total_requests > 0 else 0
    
    session_duration = (datetime.now() - session_start_time).total_seconds() / 60
    requests_per_minute = total_requests / session_duration if session_duration > 0 else 0
    
    print("\n" + "="*60)
    print("📊 CRAWLBASE SESSION STATISTICS")
    print("="*60)
    print(f"Total Requests: {total_requests}")
    print(f"Successful: {successful_requests} ({success_rate:.1f}%)")
    print(f"Failed: {total_requests - successful_requests}")
    print(f"Total Retries: {total_retries}")
    print(f"Average Retries per Request: {avg_retries:.2f}")
    print(f"Consecutive Successful: {consecutive_requests}")
    print(f"Session Duration: {session_duration:.1f} minutes")
    print(f"Request Rate: {requests_per_minute:.1f} requests/minute")
    
    # AI Optimization Status
    if consecutive_requests >= 50:
        print("🤖 AI OPTIMIZATION: ACTIVE ✅")
        print("   (50+ consecutive requests completed)")
    else:
        remaining = 50 - consecutive_requests
        print(f"🤖 AI OPTIMIZATION: PENDING ({remaining} more requests needed)")
    
    print("="*60)

# Test the enhanced function
if 'token' in locals() and token and token != 'your_token_here':
    print("🚀 Testing Enhanced Crawlbase Implementation...")
    
    # Test with existing URL
    if 'full_url' in locals():
        html = fetch_with_crawlbase(full_url, token)
        
        if html:
            print(f"\n✅ Test successful! Received {len(html):,} characters")
            print("\nFirst 300 characters:")
            # Handle both string and bytes
            if isinstance(html, bytes):
                preview = html[:300].decode('utf-8', errors='ignore') + "..."
            else:
                preview = html[:300] + "..."
            print(preview)
        else:
            print("\n❌ Test failed - no HTML received")
    
    # Print initial stats
    print_session_stats()
    
else:
    print("⚠️  Please ensure your token is loaded before testing")


# In[8]:


# Cell 7: Parse HTML to Find Price (with debugging)
print("Starting Cell 7...")

# Check if we have HTML
if 'html' not in globals():
    print("❌ 'html' variable not found. Please run Cell 6 first.")
elif html is None:
    print("❌ 'html' is None. Cell 6 didn't fetch the page successfully.")
elif len(html) == 0:
    print("❌ 'html' is empty.")
else:
    print(f"✓ HTML found: {len(html)} characters")
    
    soup = BeautifulSoup(html, 'html.parser')
    print("✓ HTML parsed with BeautifulSoup")
    
    # Find all text containing '$'
    price_texts = soup.find_all(text=lambda text: '$' in text if text else False)
    
    print(f"\nFound {len(price_texts)} elements with '$'")
    
    if len(price_texts) == 0:
        print("No dollar signs found in the HTML")
        # Let's try a different approach - look for common price classes
        price_elements = soup.find_all(['span', 'div'], class_=lambda x: x and 'price' in x.lower() if isinstance(x, str) else False)
        print(f"\nFound {len(price_elements)} elements with 'price' in class name")
    else:
        print("\nFirst 10 price-like texts:")
        count = 0
        for i, text in enumerate(price_texts):
            clean_text = text.strip()
            if clean_text and len(clean_text) < 50:
                if '$' in clean_text and any(char.isdigit() for char in clean_text):
                    count += 1
                    print(f"{count}: {clean_text}")
                    if count >= 10:
                        break
        
        if count == 0:
            print("No price-like text found (text with $ and numbers)")

print("\nCell 7 complete.")


# In[9]:


# Cell 8: Extract Lowest Fully Refundable Rate
def extract_price_from_html(html):
    """
    Extract the lowest FULLY REFUNDABLE rate from Expedia HTML
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    print("🔍 Searching for 'Fully refundable' rates...\n")
    
    # Find all text containing "Fully refundable"
    refundable_texts = soup.find_all(string=lambda text: text and 'fully refundable' in text.lower())
    
    print(f"Found {len(refundable_texts)} 'Fully refundable' mentions")
    
    prices = []
    
    for refund_text in refundable_texts:
        # Navigate up from "Fully refundable" text to find the container
        current = refund_text.parent
        
        # Go up the DOM tree to find the card/container with the price
        for _ in range(15):  # Check up to 15 levels
            if current:
                # Look for a nightly price in this container
                price_elem = current.find(string=lambda text: text and '$' in text and 'nightly' in text)
                
                if price_elem:
                    # Extract the price
                    import re
                    match = re.search(r'\$(\d+(?:,\d+)*)\s*nightly', price_elem)
                    if match:
                        price_str = match.group(1).replace(',', '')
                        price = float(price_str)
                        prices.append(price)
                        print(f"✓ Found fully refundable rate: ${price}")
                        break  # Found price for this refundable option
                
                current = current.parent
    
    if prices:
        # Remove duplicates and get the lowest
        unique_prices = list(set(prices))
        unique_prices.sort()
        lowest_price = unique_prices[0]
        
        print(f"\nUnique fully refundable rates: {unique_prices}")
        print(f"✅ Lowest fully refundable rate: ${lowest_price}")
        
        return lowest_price
    else:
        print("❌ No fully refundable rates found")
        return None

# Test the function
price = extract_price_from_html(html)

if price:
    print(f"\n✅ SUCCESS! Lowest fully refundable rate: ${price}")
else:
    print("\n❌ Could not find fully refundable rate")


# In[10]:


def scrape_hotel_rate(hotel_name, hotel_id, check_in_date, token):
    """
    Scrape rate for one hotel on one date
    Returns a dictionary with the results
    """
    
    # Calculate check-out (next day)
    check_out_date = check_in_date + timedelta(days=1)
    
    # Format dates
    check_in_str = check_in_date.strftime('%Y-%m-%d')
    check_out_str = check_out_date.strftime('%Y-%m-%d')
    
    # Build URL
    hotel_url_name = hotel_name.replace(' ', '-')
    url = f"https://www.expedia.com/New-York-Hotels-{hotel_url_name}.h{hotel_id}.Hotel-Information"
    url += f"?chkin={check_in_str}&chkout={check_out_str}"
    
    # Fetch page
    html = fetch_with_crawlbase(url, token)
    
    # Extract price
    if html:
        price = extract_price_from_html(html)
        available = price is not None
    else:
        price = None
        available = False
    
    # Return results
    return {
        'hotel_name': hotel_name,
        'hotel_id': hotel_id,
        'check_in_date': check_in_str,
        'check_out_date': check_out_str,
        'price': price,
        'is_available': available,
        'scrape_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'scrape_date': datetime.now().strftime('%Y-%m-%d')
    }

# Test with one hotel and date
test_result = scrape_hotel_rate(
    hotels[0]['name'], 
    hotels[0]['id'], 
    datetime.now() + timedelta(days=7),
    token
)

print("Scrape result:")
for key, value in test_result.items():
    print(f"  {key}: {value}")

# Room Type HTML Analysis - Analysis Code
if 'html' in globals() and html:
    print("\n🔍 ANALYZING HTML FOR ROOM TYPE PATTERNS")
    print("="*50)
    
    soup = BeautifulSoup(html, 'html.parser')
    
    # Look for common room type keywords in text
    room_keywords = ['room', 'suite', 'king', 'queen', 'standard', 'deluxe', 'junior', 'executive']
    
    print("🏨 Searching for room type mentions...")
    room_mentions = []
    
    for keyword in room_keywords:
        elements = soup.find_all(text=lambda text: text and keyword.lower() in text.lower())
        for element in elements[:3]:  # Limit to first 3 matches per keyword
            clean_text = element.strip()
            if len(clean_text) > 5 and len(clean_text) < 100:  # Reasonable length
                room_mentions.append(clean_text)
    
    # Remove duplicates and show findings
    unique_mentions = list(set(room_mentions))[:10]  # Show first 10 unique
    
    print(f"\nFound {len(unique_mentions)} potential room type mentions:")
    for i, mention in enumerate(unique_mentions, 1):
        print(f"{i}. {mention}")
    
    # Look for elements near price information
    print(f"\n💰 Analyzing elements near pricing...")
    price_elements = soup.find_all(text=lambda text: text and '$' in text and any(c.isdigit() for c in text))
    
    print(f"Found {len(price_elements)} price-related elements")
    
    # Look for parent elements of prices to find room context
    print(f"\n🔗 Examining price contexts...")
    for i, price_elem in enumerate(price_elements[:5]):  # Check first 5 prices
        parent = price_elem.parent
        if parent:
            # Look for room type info in nearby siblings or parents
            siblings = [s.get_text().strip() for s in parent.find_all() if s.get_text().strip()]
            room_context = [s for s in siblings if any(keyword in s.lower() for keyword in room_keywords)]
            
            if room_context:
                print(f"Price {i+1}: {price_elem.strip()[:30]}")
                print(f"  Room context: {room_context[:2]}")  # Show first 2 matches
    
else:
    print("❌ No HTML available for analysis. Run previous cells first.")
    


# In[11]:


# Cell 10: Scrape Multiple Dates for One Hotel
def scrape_hotel_all_dates(hotel_name, hotel_id, start_date, days_ahead, token):
    """
    Scrape rates for one hotel for multiple dates
    """
    results = []
    
    print(f"Scraping {hotel_name} for {days_ahead} days...")
    
    for day in range(days_ahead):
        check_in = start_date + timedelta(days=day)
       
              
        print(f"  Day {day+1}/{days_ahead}: {check_in.strftime('%Y-%m-%d')}", end='')
        
        # Use the scrape_hotel_rate function which uses the correct extraction
        result = scrape_hotel_rate(hotel_name, hotel_id, check_in, token)
        results.append(result)
        
        if result['is_available']:
            print(f" - ${result['price']}")
        else:
            print(" - Not available")
    
    return results

# Test with one hotel for next 5 days
test_results = scrape_hotel_all_dates(
    hotels[0]['name'],
    hotels[0]['id'],
    datetime.now(),
    5,  # Just 5 days for testing
    token
)

# Convert to DataFrame for easy viewing
df_test = pd.DataFrame(test_results)
df_test


# In[12]:


# Enhanced Cell 11: Optimized Parallel Scraping for Crawlbase AI
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import threading
import time

# Thread-safe print function
print_lock = threading.Lock()

def safe_print(message):
    with print_lock:
        print(message)

def scrape_hotel_date_pair_enhanced(hotel, check_in_date, token):
    """Enhanced scraping with better error handling and monitoring"""
    try:
        result = scrape_hotel_rate(hotel['name'], hotel['id'], check_in_date, token)
        
        # Enhanced status reporting
        if result['is_available']:
            status = f"✅ ${result['price']}"
        else:
            status = "❌ N/A"
            
        safe_print(f"{hotel['name'][:20]:<20} | {check_in_date.strftime('%Y-%m-%d')} | {status}")
        return result
        
    except Exception as e:
        error_msg = str(e)[:50]
        safe_print(f"💥 {hotel['name'][:20]:<20} | {check_in_date.strftime('%Y-%m-%d')} | ERROR: {error_msg}")
        
        return {
            'hotel_name': hotel['name'],
            'hotel_id': hotel['id'],
            'check_in_date': check_in_date.strftime('%Y-%m-%d'),
            'check_out_date': (check_in_date + timedelta(days=1)).strftime('%Y-%m-%d'),
            'price': None,
            'is_available': False,
            'scrape_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'scrape_date': datetime.now().strftime('%Y-%m-%d'),
            'error': error_msg
        }

def scrape_all_hotels_optimized(hotels, days_ahead=30, max_workers=25, batch_size=50):
    """
    Optimized scraping that maintains steady request volume for AI optimization
    
    Args:
        hotels: List of hotel dictionaries
        days_ahead: Number of days to scrape
        max_workers: Maximum parallel workers (reduced for steadier volume)
        batch_size: Process requests in batches for better AI optimization
    """
    
    # Get token
    token = get_crawlbase_token()
    if not token or token == 'your_token_here':
        print("❌ Error: Please set your Crawlbase token in hip_config.txt")
        return None
    
    # Reset monitoring for this session
    global request_history, consecutive_requests, session_start_time
    request_history.clear()
    consecutive_requests = 0
    session_start_time = datetime.now()
    
    # Starting date (today)
    start_date = datetime.now()
    
    # Create all hotel/date combinations
    tasks = []
    for hotel in hotels:
        for day in range(days_ahead):
            check_in = start_date + timedelta(days=day)
            tasks.append((hotel, check_in, token))
    
    total_tasks = len(tasks)
    print("🚀 ENHANCED HIP RATE SCRAPING")
    print("="*50)
    print(f"Hotels: {len(hotels)}")
    print(f"Days: {days_ahead}")
    print(f"Total requests: {total_tasks}")
    print(f"Parallel workers: {max_workers}")
    print(f"Batch size: {batch_size}")
    print(f"Estimated time: {(total_tasks / max_workers * 8) / 60:.1f} minutes")
    print("="*50)
    print(f"{'Hotel':<20} | {'Date':<10} | {'Result'}")
    print("-" * 50)
    
    all_results = []
    completed_count = 0
    
    # Process in batches for steady request volume
    for batch_start in range(0, len(tasks), batch_size):
        batch_end = min(batch_start + batch_size, len(tasks))
        batch_tasks = tasks[batch_start:batch_end]
        
        batch_num = (batch_start // batch_size) + 1
        total_batches = (len(tasks) + batch_size - 1) // batch_size
        
        safe_print(f"\n📦 Processing Batch {batch_num}/{total_batches} ({len(batch_tasks)} requests)")
        
        # Execute batch in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit batch tasks
            futures = []
            for task in batch_tasks:
                future = executor.submit(scrape_hotel_date_pair_enhanced, task[0], task[1], task[2])
                futures.append(future)
            
            # Process completed tasks in batch
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    all_results.append(result)
                    completed_count += 1
                    
                except Exception as e:
                    safe_print(f"💥 Batch task failed: {e}")
                    completed_count += 1
        
        # Progress update
        progress_pct = (completed_count / total_tasks) * 100
        safe_print(f"📊 Batch {batch_num} complete. Overall progress: {completed_count}/{total_tasks} ({progress_pct:.1f}%)")
        
        # Small delay between batches for steady volume
        if batch_end < len(tasks):  # Not the last batch
            time.sleep(2)
    
    # Final statistics
    print("\n" + "="*60)
    print("✅ SCRAPING COMPLETE!")
    print("="*60)
    
    # Print session statistics
    print_session_stats()
    
    # Results summary
    available_results = [r for r in all_results if r['is_available']]
    print(f"\n📈 RESULTS SUMMARY:")
    print(f"Total scraped: {len(all_results)}")
    print(f"Available rates: {len(available_results)}")
    print(f"Success rate: {len(available_results)/len(all_results)*100:.1f}%")
    
    if available_results:
        prices = [r['price'] for r in available_results if r['price']]
        if prices:
            print(f"Price range: ${min(prices):.0f} - ${max(prices):.0f}")
            print(f"Average price: ${sum(prices)/len(prices):.0f}")
    
    return all_results

# Example usage with enhanced monitoring
print("🔧 Enhanced scraping functions loaded!")
print("💡 Use scrape_all_hotels_optimized() for best AI optimization results")
print("📊 Monitor request_history and print_session_stats() for performance insights")

results = scrape_all_hotels_optimized(hotels, days_ahead=30)


# In[13]:


# Cell 12: Save Results to CSV and Upload to Azure
from azure.storage.blob import BlobServiceClient

def save_results_to_csv(results, filename=None):
    """
    Save scraping results to CSV file and upload to Azure
    """
    
    if not results:
        print("❌ No results to save")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'HIP2_rates_{timestamp}.csv'
    
    # Save to CSV locally
    df.to_csv(filename, index=False)
    print(f"✅ Saved {len(df)} rows to {filename}")
    
    # Also save as .txt to avoid Excel auto-conversion
    txt_filename = filename.replace('.csv', '.txt')
    df.to_csv(txt_filename, index=False)
    print(f"✅ Also saved as {txt_filename}")
    
    # Upload to Azure
    try:
        with open('azure_config.txt', 'r') as f:
            connection_string = f.read().strip()
            
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_name = "hip-hotel-rates"
        
        # Upload CSV to Azure
        csv_data = df.to_csv(index=False)
        blob_client = blob_service_client.get_blob_client(
            container=container_name, 
            blob=f"rates/hip2/{filename}"
        )
        blob_client.upload_blob(csv_data, overwrite=True)
        print(f"☁️  Uploaded to Azure: {container_name}/rates/{filename}")
        
    except Exception as e:
        print(f"⚠️  Azure upload failed (data saved locally): {e}")
    
    # Show summary statistics
    print("\n📊 Summary Statistics:")
    print(f"Total hotels: {df['hotel_name'].nunique()}")
    print(f"Date range: {df['check_in_date'].min()} to {df['check_in_date'].max()}")
    print(f"Available rates: {df['is_available'].sum()} out of {len(df)}")
    
    if df['price'].notna().any():
        print(f"\n💰 Price Statistics:")
        print(f"Average price: ${df['price'].mean():.2f}")
        print(f"Lowest price: ${df['price'].min():.2f}")
        print(f"Highest price: ${df['price'].max():.2f}")
    
    return df.head(10)

# Save our test results
if results:
    df_results = save_results_to_csv(results)
    display(df_results)


# In[14]:


# Cell 13: Analyze Results (Enhanced)
# Let's visualize what we collected
if 'results' in globals() and results and len(results) > 0:
    df = pd.DataFrame(results)
    
    print(f"📊 Analysis of {len(df)} scraping results\n")
    
    # Show availability summary
    available_count = df['is_available'].sum()
    print(f"Available rates: {available_count} out of {len(df)} ({available_count/len(df)*100:.1f}%)")
    
    # Create a pivot table to see prices by hotel and date
    if df['price'].notna().any():
        # Ensure we have only one price per hotel/date (the lowest)
        df_clean = df.groupby(['hotel_name', 'check_in_date']).agg({
            'price': 'min',
            'is_available': 'first'
        }).reset_index()
        
        pivot = df_clean.pivot_table(
            values='price', 
            index='hotel_name', 
            columns='check_in_date', 
            aggfunc='first'
        )
        
        print("\n💰 Price Matrix (Hotel x Date):")
        display(pivot)
        
        # Quick visualization
        import matplotlib.pyplot as plt
        
        # Plot prices over time for each hotel
        plt.figure(figsize=(12, 8))
        
        for hotel in df_clean['hotel_name'].unique():
            hotel_data = df_clean[df_clean['hotel_name'] == hotel]
            available_data = hotel_data[hotel_data['price'].notna()]
            
            if not available_data.empty:
                # Sort by date for smooth line
                available_data = available_data.sort_values('check_in_date')
                
                plt.plot(
                    pd.to_datetime(available_data['check_in_date']), 
                    available_data['price'], 
                    marker='o', 
                    label=hotel,
                    linewidth=2,
                    markersize=6
                )
        
        plt.xlabel('Check-in Date')
        plt.ylabel('Price ($)')
        plt.title('Hotel Rates Over Time (Lowest Fully Refundable Rate per Day)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Set y-axis to start at 0 for better comparison
        plt.ylim(bottom=0)
        
        plt.tight_layout()
        plt.show()
    else:
        print("\n❌ No price data available to visualize")
        print("This is expected if all dates are within 72-hour booking window")
        
        # Show availability by date
        availability_by_date = df.groupby('check_in_date')['is_available'].sum()
        print(f"\nAvailability by date:")
        print(availability_by_date)
else:
    print("❌ No results found. Please run the scraping cells first.")

# NaN Analysis KPI
nan_count = pivot.isna().sum().sum()  # Total NaN values in price matrix
total_cells = pivot.shape[0] * pivot.shape[1]  # Total matrix cells
data_completeness = ((total_cells - nan_count) / total_cells) * 100

print(f"\n📊 DATA COMPLETENESS KPI:")
print(f"Total matrix cells: {total_cells}")
print(f"NaN (missing) values: {nan_count}")
print(f"Data completeness: {data_completeness:.1f}%")

# HIP2 SCRIPT END TIMER
script_end_time = datetime.now()
total_runtime = script_end_time - script_start_time
print(f"\n🏁 HIP2 Script Completed: {script_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"⏱️  Total Runtime: {total_runtime}")
print(f"📊 Runtime in minutes: {total_runtime.total_seconds()/60:.1f} minutes")


# In[ ]:




