#!/usr/bin/env python3
"""
Parse KLHK KMZ/KML description fields to extract attributes
Converts HTML table in description to proper GeoDataFrame columns
"""

import geopandas as gpd
import pandas as pd
from bs4 import BeautifulSoup
import re

def parse_description(desc):
    """Parse HTML description table to dictionary"""
    if pd.isna(desc) or not desc:
        return {}

    try:
        soup = BeautifulSoup(desc, 'html.parser')
        table = soup.find('table')

        if not table:
            return {}

        data = {}
        rows = table.find_all('tr')

        for row in rows[1:]:  # Skip header row
            cells = row.find_all('td')
            if len(cells) == 2:
                field_name = cells[0].get_text(strip=True)
                field_value = cells[1].get_text(strip=True)
                data[field_name] = field_value

        return data
    except Exception as e:
        print(f"Error parsing description: {e}")
        return {}

def process_klhk_geojson(input_file, output_file):
    """
    Process KLHK GeoJSON from KMZ export
    Extracts attributes from description and creates clean GeoDataFrame
    """
    print(f"Loading {input_file}...")
    gdf = gpd.read_file(input_file)

    print(f"Total features: {len(gdf)}")
    print(f"Parsing descriptions...")

    # Parse all descriptions
    parsed_data = []
    for idx, row in gdf.iterrows():
        attrs = parse_description(row['description'])
        attrs['geometry'] = row['geometry']
        parsed_data.append(attrs)

    # Create new GeoDataFrame with parsed attributes
    gdf_clean = gpd.GeoDataFrame(parsed_data, crs=gdf.crs)

    print(f"\nExtracted columns: {list(gdf_clean.columns)}")
    print(f"Sample record:")
    print(gdf_clean.iloc[0].drop('geometry').to_dict())

    # Save clean version
    print(f"\nSaving to {output_file}...")
    gdf_clean.to_file(output_file, driver='GeoJSON')

    print(f"✅ Saved {len(gdf_clean)} features with clean attributes")

    # Summary statistics
    if 'Kode Provinsi' in gdf_clean.columns:
        print(f"\nKode Provinsi distribution:")
        print(gdf_clean['Kode Provinsi'].value_counts())

    if 'ID Penutupan Lahan Tahun 2024' in gdf_clean.columns:
        print(f"\nPL2024 class distribution:")
        print(gdf_clean['ID Penutupan Lahan Tahun 2024'].value_counts().head(10))

    return gdf_clean

if __name__ == "__main__":
    import sys

    # Default files
    input_file = 'data/klhk/KLHK_PL2024_Jambi_batch1.geojson'
    output_file = 'data/klhk/KLHK_PL2024_Jambi_batch1_clean.geojson'

    # Allow command line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]

    gdf = process_klhk_geojson(input_file, output_file)

    print(f"\n✅ Processing complete!")
    print(f"Output: {output_file}")
