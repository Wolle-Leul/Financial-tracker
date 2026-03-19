# Financial-tracker Dashboard (Streamlit)

This project is a personal finance dashboard built with Streamlit. It lets you:
1. Upload a bank statement PDF
2. Parse transactions from the PDF and store them in a database (Postgres/Supabase)
3. Auto-map transactions to expense categories/subcategories (keyword rules)
4. Compute cash-flow metrics for a selected month/year based on a salary window

## Local setup

1. Install dependencies:
   - `pip install -r requirements.txt`
   - Note: the calendar uses the optional `holidays` library for Polish holiday markers. If it isn't installed, the dashboard will still run, but the calendar will show no holiday dots.
2. Set environment variables (recommended):
   - `DB_URL` (example: `postgresql+psycopg://user:pass@host:5432/dbname`)
   - `PASSWORD_HASH` (bcrypt hash string)
3. Run DB migrations:
   - `python -m finance_tracker.db.migrate`
4. Start the dashboard:
   - `streamlit run app.py`

## Streamlit Cloud: secrets

In Streamlit Cloud, configure secrets as shown in [`./.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example).

Required secrets:
- `db_url`: Postgres/Supabase connection string
- `password_hash`: bcrypt hash of your password

Recommended optional secrets:
- `holiday_country` (default: `Poland`)
- `salary_day_of_month` (default: `10`)
- `target_ratio` (default: `0.45`)

## Create a `password_hash` (bcrypt)

Run this once locally and copy the printed hash into Streamlit secrets:

```python
import bcrypt

password = "your-password"
hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
print(hashed.decode("utf-8"))
```

## How imports and category mapping work

- On first run (after migrations), the app seeds demo categories/subcategories into the DB, including keyword rules used for auto-mapping.
- When you upload a PDF:
  - The app extracts text with `PyPDF2`
  - Uses regex heuristics to detect `(date, amount, description)` rows
  - Stores an `imports` record and the parsed `transactions`
  - Attempts keyword-based mapping into `category_id` and `subcategory_id`
- If some transactions remain uncategorized, use the sidebar expander:
  - `Manual category override (uncategorized)`

## Notes

- Metrics are computed for the selected month/year using a salary window derived from the salary day rule and holidays.
- The dashboard shows an empty-state message when there are no transactions in the selected salary window.

