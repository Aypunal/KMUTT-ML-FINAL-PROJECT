scrape_real_imp.py is run for create storage for scrape and will scrape picture as TARGET_TOTAL on topic and specific site in queries(can also just add topic no need to always add specific site).(still have like 0.66% picture that not real life  picture ex. infographic or something like that)
and the make_7_day_pic.py is random pic DAYS * IMAGES_PER_DAY to store in 1 folder and DAYS subfolders it also contain log for picture that already use so if you run later it will not get the picture that already use.
https://pypi.org/project/duckduckgo-search/#duckduckgo-search-operators
library for scrape that I use for clrarification on some detail.

# How to run Web Application
## Requirements
```
1.Node.js
2.Python
```
## Backend
```
1. cd backend
2. pip install -r requirements.txt
3. python api.py
```

## Frontend
```
1. npm install
2. update api endpoint on page.jsx (your backend api endpoint, will do .env later)
3. npm run build
4. npm start
```