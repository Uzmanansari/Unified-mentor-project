# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 23:25:03 2024

@author: uzman
"""
# Re-import necessary libraries
import pandas as pd
import numpy as np

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Load dataset
FDI = pd.read_csv('FDI data.csv')

# Display basic details
print("Dataset Shape:", FDI.shape)
print("Dataset Columns:", FDI.columns)
FDI.info()

# Step 1: Check for missing and duplicate values
print("\nMissing Values:\n", FDI.isnull().sum())
print("\nDuplicate Rows Count:", FDI.duplicated().sum())

# Step 2: Define years and exchange rates (from RBI)
Years = ['2000-01', '2001-02', '2002-03', '2003-04', '2004-05',
         '2005-06', '2006-07', '2007-08', '2008-09', '2009-10',
         '2010-11', '2011-12', '2012-13', '2013-14', '2014-15',
         '2015-16', '2016-17']

Exchange_Rates = [45.68, 47.69, 48.39, 45.95, 44.93, 44.27, 45.24, 40.26,
                  45.99, 47.44, 45.56, 47.92, 54.40, 60.50, 61.14, 65.46, 67.07]

# Step 3: Convert FDI values from USD to INR
def convert_to_inr(df, years, rates):
    for year in years:
        df[year] = df[year] * rates[years.index(year)] / 10  # Convert to ₹ Crores
    return df

FDI_INR = FDI.copy()
FDI_INR = convert_to_inr(FDI_INR, Years, Exchange_Rates)

# Step 4: Unpivot (reshape) data from wide to long format
FDI_USD_long = pd.melt(FDI, id_vars=['Sector'], value_vars=Years,
                       var_name='Year', value_name='FDI_USD_Million')

FDI_INR_long = pd.melt(FDI_INR, id_vars=['Sector'], value_vars=Years,
                       var_name='Year', value_name='FDI_INR_Crores')

# Merge USD and INR data
FDI_merged = FDI_INR_long.merge(FDI_USD_long, on=['Sector', 'Year'], how='left')

# Step 5: Shorten long sector names for clarity
sector_replacements = {
    "CONSTRUCTION DEVELOPMENT: Townships, housing, built-up infrastructure and construction-development projects": "CONSTRUCTION DEVELOPMENT",
    "SERVICES SECTOR (Fin.,Banking,Insurance,Non Fin/Business,Outsourcing,R&D,Courier,Tech. Testing and Analysis, Other)": "SERVICES SECTOR",
    "TEA AND COFFEE (PROCESSING & WAREHOUSING COFFEE & RUBBER)": "TEA AND COFFEE"
}
FDI_merged['Sector'] = FDI_merged['Sector'].replace(sector_replacements)

# Step 6: Calculate total FDI inflow per sector (2000-2017)
Sectorwise_fdi = FDI_merged.groupby('Sector')[['FDI_INR_Crores', 'FDI_USD_Million']].sum()
Sectorwise_fdi['Year'] = '2000-17'
Sectorwise_fdi['% of Total Inflows'] = (Sectorwise_fdi['FDI_INR_Crores'] /
                                        Sectorwise_fdi['FDI_INR_Crores'].sum()) * 100

# Step 7: Sort by total FDI inflow
Sectorwise_fdi = Sectorwise_fdi.sort_values(by='FDI_INR_Crores', ascending=False)

# Step 8: Export cleaned and aggregated data to CSV
FDI_merged.to_csv('clean_FDI.csv', index=False)
Sectorwise_fdi.to_csv('sectorwise_FDI.csv', index=False)

# Display the cleaned sector-wise FDI inflows
print("\nSector-Wise FDI Inflows (2000-2017):")
print(Sectorwise_fdi)


