import requests
import zipfile
import io
import psycopg2
from psycopg2.extras import execute_values
import os
import gc


DATABASE_URL = os.environ.get("DATABASE_URL")

def run_gdelt_update():
    print("Fetching latest GDELT update URL...")
    with requests.get("http://data.gdeltproject.org/gdeltv2/lastupdate.txt") as response:
        file_url = response.text.strip().split("\n")[0].split()[2]

    data_to_insert = []
    
    with requests.get(file_url) as zip_response:
        with zipfile.ZipFile(io.BytesIO(zip_response.content)) as zip_file:
            with zip_file.open(zip_file.namelist()[0]) as csv_file:
                for line in csv_file:
                    decoded = line.decode("utf-8").strip()
                    if decoded:
                        row = decoded.split("\t")
                        data_to_insert.append([val if val != "" else None for val in row])

    print(f"Connecting to cloud database to insert {len(data_to_insert)} rows...")
    try:
    
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()

        print("Emptying previous batch from public.gdelt_events...")
        cursor.execute("TRUNCATE TABLE public.gdelt_events;")

        
        insert_query = "INSERT INTO public.gdelt_events VALUES %s ON CONFLICT (GLOBALEVENTID) DO NOTHING"
        execute_values(cursor, insert_query, data_to_insert)
        
        conn.commit()
        print("Success! Cloud database updated.")
    except Exception as e:
        print(f"Database error: {e}")
    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()
            
    data_to_insert.clear()
    gc.collect()


if __name__ == "__main__":
    run_gdelt_update()
