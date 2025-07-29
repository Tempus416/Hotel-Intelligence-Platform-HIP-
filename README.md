Hotel Intelligence Platform (HIP)
Crawlbase edition
=================================

Overview
--------
Hotel Intelligence Platform (HIP) is a Python-based web scraping and data intelligence tool built to extract hotel room availability and pricing data from Expedia for a selected group of luxury properties. The project supports strategic rate tracking over a rolling 30-day window and stores results in structured formats for business intelligence, forecasting, or competitive analysis.

Features
--------
- Scrapes real-time hotel pricing & availability from Expedia  
- Supports 30-day rolling rate lookups across multiple hotels  
- Handles pagination, date formatting, and anti-bot protection using Crawlbase  
- Runs in parallel using multithreading for high speed  
- Outputs data to .csv, .txt, and optionally .xlsx for Excel compatibility  
- Easy configuration via hip_config.txt for Crawlbase token  
- Ready for integration with BI tools

Technologies Used
-----------------
- requests, BeautifulSoup – web scraping
- pandas – data wrangling
- datetime – date generation
- concurrent.futures – multithreaded scraping
- Crawlbase – anti-bot proxy API
- Jupyter Notebook – development environment
- Azure Blob (optional) – for future cloud storage expansion

Project Structure
-----------------
Hotel Intelligence Platform (HIP).ipynb  # Main notebook  
hip_config.txt                           # Configuration file (stores token)  
HIP_rates_YYYYMMDD_HHMMSS.csv           # Output data (CSV)  
HIP_rates_YYYYMMDD_HHMMSS.txt           # Output data (TXT mirror)  
/images/                                 # (Optional) Visualization exports

Setup Instructions
------------------
1. Install required packages:
   pip install pandas requests beautifulsoup4 azure-storage-blob

2. Clone this repo and open the notebook:
   git clone https://github.com/yourusername/HIP-scraper.git  
   cd HIP-scraper  
   jupyter notebook

3. Create the config file:
   - Run the notebook’s cell that generates hip_config.txt.
   - Paste in your Crawlbase JavaScript token:
     CRAWLBASE_TOKEN=your_real_token_here

4. Run the scraper:
   - Define hotels to track (already pre-loaded in the notebook).
   - Adjust scraping window if desired (default is 30 days).
   - Run the notebook. Results will save automatically.

Output
------
The script saves:
- .csv — for general use and upload to databases like Snowflake
- .txt — avoids Excel auto-formatting issues

Sample columns:
hotel_name, hotel_id, check_in_date, price, is_available, scrape_timestamp, error

Optional Enhancements
---------------------
- Power BI or Tableau connection to analyze exported .csv files  
- Add ML model for dynamic pricing alerts

Security
--------
- Token is stored locally in a .txt config file and never exposed in code.
- Avoid pushing your token to GitHub by adding this to .gitignore:
  hip_config.txt

License
-------
MIT License

Author
------
Developed by Brian La Monica  
LinkedIn: https://www.linkedin.com/in/brianlamonica/  
GitHub: https://github.com/Tempus416
