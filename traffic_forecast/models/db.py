"""
Database initialization script.
"""

import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from traffic_forecast import PROJECT_ROOT

load_dotenv()

def init_db():
 db_url = os.getenv('POSTGRES_DSN', 'postgresql://user:pass@localhost:5432/traffic')
 engine = create_engine(db_url)

 schema_path = PROJECT_ROOT / 'infra' / 'sql' / 'schema.sql'
 with schema_path.open('r', encoding='utf-8') as f:
 sql = f.read()

 with engine.connect() as conn:
 conn.execute(text(sql))
 conn.commit()

 print("Database initialized.")

if __name__ == "__main__":
 init_db()
