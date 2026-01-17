"""atDiesel demand indicator metadata for FRED series selection."""

from __future__ import annotations

DIESEL_DEMAND_GROUPS = [
    (
        "Direct Diesel Market (Target & Supply)",
        [
            ("GASDESW", "US No 2 Diesel Retail Prices"),
            ("DDFUELUSGULF", "Spot Price: Gulf Coast Ultra-Low Sulfur Diesel"),
            ("WPU057303", "PPI: #2 Diesel Fuel"),
            ("WPU057", "PPI: Refined Petroleum Products"),
        ],
    ),
    (
        "Freight & Transportation (Primary Drivers)",
        [
            ("TSIFRGHT", "Freight Transportation Services Index"),
            ("TRUCKD11", "Truck Tonnage Index"),
            ("RAILFRTCARLOADSD11", "Rail Freight Carloads"),
            ("RAILFRTINTERMODALD11", "Rail Freight Intermodal Traffic"),
            ("VMTD11", "Vehicle Miles Traveled (Seasonally Adjusted)"),
            ("HTRUCKSSA", "Motor Vehicle Retail Sales: Heavy Weight Trucks"),
            ("TOTALSA", "Total Vehicle Sales"),
            ("CES4348400001", "All Employees: Truck Transportation"),
            ("PCU484121484121P", "PPI: General Freight Trucking, Long-Distance Truckload"),
            ("PCU484110484110P", "PPI: General Freight Trucking, Local"),
            ("CUSR0000SETB01", "CPI: Gasoline (All Types)"),
            ("PCU482482", "PPI: Rail Transportation"),
            ("PCU481481", "PPI: Air Transportation"),
            ("PCU483483", "PPI: Water Transportation"),
        ],
    ),
    (
        "Manufacturing & Industrial (Leading Indicators)",
        [
            ("INDPRO", "Industrial Production: Total Index"),
            ("IPMAN", "Industrial Production: Manufacturing (NAICS)"),
            ("IPGMFN", "Industrial Production: Manufacturing (SIC)"),
            ("TCU", "Capacity Utilization: Total Industry"),
            ("DGORDER", "Manufacturers' New Orders: Durable Goods"),
            ("NEWORDER", "Manufacturers' New Orders: Nondefense Capital Goods"),
            ("AMTMNO", "Value of Manufacturers' New Orders for All Manufacturing Industries"),
            ("BUSINV", "Total Business Inventories"),
            ("ISRATIO", "Total Business: Inventories to Sales Ratio"),
            ("WHLSLRIRSA", "Wholesale Inventory-to-Sales Ratio"),
            ("IPMINE", "Industrial Production: Mining"),
            ("IPDCONGD", "Industrial Production: Durable Consumer Goods"),
            ("IPNCONGD", "Industrial Production: Nondurable Consumer Goods"),
            ("IPB50001N", "Industrial Production: Construction Supplies"),
            ("IPMAT", "Industrial Production: Materials"),
            ("IPBUSEQ", "Industrial Production: Business Equipment"),
            ("AWHAEMAN", "Average Weekly Hours of All Employees, Manufacturing"),
        ],
    ),
    (
        "Regional Fed Surveys (Leading Indicators)",
        [
            ("GACDFSA066MSFRBPHI", "Philly Fed Outlook: General Activity"),
            ("GACDISA066MSFRBNY", "Empire State Survey: New Orders"),
            ("CFNAI", "Chicago Fed National Activity Index"),
            ("CFNAIMA3", "Chicago Fed National Activity Index: 3 Month Moving Average"),
        ],
    ),
    (
        "Construction & Housing (Off-Road Demand)",
        [
            ("HOUST", "Housing Starts: Total New Privately Owned Housing Units"),
            ("PERMIT", "New Private Housing Units Authorized by Building Permits"),
            ("TTLCONS", "Total Construction Spending"),
            ("PNRESCONS", "Private Nonresidential Construction Spending"),
            ("PRRESCONS", "Private Residential Construction Spending"),
            ("USCONS", "All Employees: Construction"),
            ("WPUSI012011", "PPI: Construction Materials"),
            ("UNDCONTSA", "Housing Units Under Construction"),
        ],
    ),
    (
        "Macro-Economic (Broad Demand)",
        [
            ("GDPC1", "Real Gross Domestic Product"),
            ("GDP", "Gross Domestic Product"),
            ("GPDIC1", "Real Gross Private Domestic Investment"),
            ("A939RX0Q048SBEA", "Real Government Consumption Expenditures"),
            ("RSAFS", "Advance Retail Sales: Retail and Food Services"),
            ("RSXFS", "Advance Retail Sales: Retail (Excl. Food Services)"),
            ("TOTRESNS", "Total Business: Retail and Food Services Sales"),
            ("ECOMSA", "E-Commerce Retail Sales"),
            ("PCE", "Personal Consumption Expenditures"),
            ("DSPIC96", "Real Disposable Personal Income"),
            ("BOPGSTB", "Trade Balance: Goods and Services"),
            ("IMP0015", "Real Imports of Goods and Services"),
            ("EXP0015", "Real Exports of Goods and Services"),
        ],
    ),
    (
        "Labor Market (Economic Health)",
        [
            ("PAYEMS", "All Employees, Total Nonfarm"),
            ("UNRATE", "Unemployment Rate"),
            ("CIVPART", "Civilian Labor Force Participation Rate"),
            ("CEU4300000001", "All Employees: Transportation and Warehousing"),
            ("AHETPI", "Average Hourly Earnings of All Employees"),
            ("JTSJOL", "Job Openings: Total Nonfarm"),
            ("ICSA", "Initial Claims"),
        ],
    ),
    (
        "Prices, Inflation & Financial (Signals)",
        [
            ("CPIAUCSL", "Consumer Price Index: All Items"),
            ("CPILFESL", "Consumer Price Index: All Items Less Food and Energy"),
            ("PCEPILFE", "PCE Excluding Food and Energy (Core PCE)"),
            ("PPIACO", "Producer Price Index: All Commodities"),
            ("WPU01", "PPI: Farm Products"),
            ("DCOILWTICO", "Spot Crude Oil Price: WTI"),
            ("DCOILBRENTEU", "Spot Crude Oil Price: Brent"),
            ("DTWEXBGS", "Trade Weighted U.S. Dollar Index"),
            ("FEDFUNDS", "Federal Funds Effective Rate"),
            ("GS10", "10-Year Treasury Constant Maturity Rate"),
            ("T10Y2Y", "10-Year Treasury Constant Maturity Minus 2-Year Treasury"),
            ("BAMLH0A0HYM2", "ICE BofA US High Yield Index Option-Adjusted Spread"),
            ("NFCI", "National Financial Conditions Index"),
            ("M2SL", "M2 Money Stock"),
            ("UMCSENT", "University of Michigan: Consumer Sentiment"),
            ("MICH", "University of Michigan: Inflation Expectations"),
        ],
    ),
]

DIESEL_DEMAND_IDS = [
    series_id for _, items in DIESEL_DEMAND_GROUPS for series_id, _ in items
]

DIESEL_DEMAND_META = {
    series_id: {"label": label, "group": group}
    for group, items in DIESEL_DEMAND_GROUPS
    for series_id, label in items
}

EIA_COVARIATE_GROUPS = [
    (
        "STEO Macro (US)",
        [
            ("GDPQXUS", "GDPQXUS"),
            ("GDPQXUS_PCT", "GDPQXUS_PCT"),
            ("GDPDIUS", "GDPDIUS"),
            ("GDPDIUS_PCT", "GDPDIUS_PCT"),
            ("YD87OUS", "YD87OUS"),
            ("YD87OUS_PCT", "YD87OUS_PCT"),
            ("ZOMNIUS", "ZOMNIUS"),
            ("ZOMNIUS_PCT", "ZOMNIUS_PCT"),
            ("CONSRUS", "CONSRUS"),
            ("I87RXUS", "I87RXUS"),
            ("KRDRXUS", "KRDRXUS"),
            ("GOVXRUS", "GOVXRUS"),
            ("TREXRUS", "TREXRUS"),
            ("TRIMRUS", "TRIMRUS"),
            ("EMNFPUS", "EMNFPUS"),
            ("XRUNR", "XRUNR"),
            ("HSTCXUS", "HSTCXUS"),
            ("ZOTOIUS", "ZOTOIUS"),
            ("ZO311IUS", "ZO311IUS"),
            ("ZO322IUS", "ZO322IUS"),
            ("ZO324IUS", "ZO324IUS"),
            ("ZO325IUS", "ZO325IUS"),
            ("ZO327IUS", "ZO327IUS"),
            ("ZO331IUS", "ZO331IUS"),
            ("QSIC_CL", "QSIC_CL"),
            ("QSIC_DF", "QSIC_DF"),
            ("QSIC_EL", "QSIC_EL"),
            ("QSIC_NG", "QSIC_NG"),
            ("CICPIUS", "CICPIUS"),
            ("WPCPIUS", "WPCPIUS"),
            ("WP57IUS", "WP57IUS"),
            ("MVVMPUS", "MVVMPUS"),
            ("RSPRPUS", "RSPRPUS"),
            ("RDPRPUS", "RDPRPUS"),
            ("BDRIPUS", "BDRIPUS"),
            ("EOPRPUS", "EOPRPUS"),
            ("BDPRPUS", "BDPRPUS"),
            ("OBPRPUS", "OBPRPUS"),
            ("EONIPUS", "EONIPUS"),
            ("BDNIPUS", "BDNIPUS"),
            ("RDNIPUS", "RDNIPUS"),
            ("OBNIPUS", "OBNIPUS"),
        ],
    ),
    (
        "STEO Diesel Fundamentals",
        [
            ("D2RPUUS", "Diesel Retail Price (US Avg)"),
            ("D2WHPUUS", "Diesel Wholesale Price (US Avg)"),
        ],
    ),
    (
        "STEO Regions (CGSP)",
        [
            ("CGSP_NEC", "CGSP_NEC"),
            ("CGSP_MAC", "CGSP_MAC"),
            ("CGSP_ENC", "CGSP_ENC"),
            ("CGSP_WNC", "CGSP_WNC"),
            ("CGSP_SAC", "CGSP_SAC"),
            ("CGSP_ESC", "CGSP_ESC"),
            ("CGSP_WSC", "CGSP_WSC"),
            ("CGSP_MTN", "CGSP_MTN"),
            ("CGSP_PAC", "CGSP_PAC"),
        ],
    ),
    (
        "STEO Regions (IPMFG)",
        [
            ("IPMFG_NEC", "IPMFG_NEC"),
            ("IPMFG_MAC", "IPMFG_MAC"),
            ("IPMFG_ENC", "IPMFG_ENC"),
            ("IPMFG_WNC", "IPMFG_WNC"),
            ("IPMFG_SAC", "IPMFG_SAC"),
            ("IPMFG_ESC", "IPMFG_ESC"),
            ("IPMFG_WSC", "IPMFG_WSC"),
            ("IPMFG_MTN", "IPMFG_MTN"),
            ("IPMFG_PAC", "IPMFG_PAC"),
        ],
    ),
    (
        "STEO Regions (CYRPIC)",
        [
            ("CYRPIC_NEC", "CYRPIC_NEC"),
            ("CYRPIC_MAC", "CYRPIC_MAC"),
            ("CYRPIC_ENC", "CYRPIC_ENC"),
            ("CYRPIC_WNC", "CYRPIC_WNC"),
            ("CYRPIC_SAC", "CYRPIC_SAC"),
            ("CYRPIC_ESC", "CYRPIC_ESC"),
            ("CYRPIC_WSC", "CYRPIC_WSC"),
            ("CYRPIC_MTN", "CYRPIC_MTN"),
            ("CYRPIC_PAC", "CYRPIC_PAC"),
        ],
    ),
    (
        "STEO Regions (QHALLC)",
        [
            ("QHALLC_NEC", "QHALLC_NEC"),
            ("QHALLC_MAC", "QHALLC_MAC"),
            ("QHALLC_ENC", "QHALLC_ENC"),
            ("QHALLC_WNC", "QHALLC_WNC"),
            ("QHALLC_SAC", "QHALLC_SAC"),
            ("QHALLC_ESC", "QHALLC_ESC"),
            ("QHALLC_WSC", "QHALLC_WSC"),
            ("QHALLC_MTN", "QHALLC_MTN"),
            ("QHALLC_PAC", "QHALLC_PAC"),
        ],
    ),
    (
        "STEO Regions (EE)",
        [
            ("EE_NEC", "EE_NEC"),
            ("EE_MAC", "EE_MAC"),
            ("EE_ENC", "EE_ENC"),
            ("EE_WNC", "EE_WNC"),
            ("EE_SAC", "EE_SAC"),
            ("EE_ESC", "EE_ESC"),
            ("EE_WSC", "EE_WSC"),
            ("EE_MTN", "EE_MTN"),
            ("EE_PAC", "EE_PAC"),
        ],
    ),
    (
        "STEO ZWHD",
        [
            ("ZWHDPUS", "ZWHDPUS"),
            ("ZWHD_NEC", "ZWHD_NEC"),
            ("ZWHD_MAC", "ZWHD_MAC"),
            ("ZWHD_ENC", "ZWHD_ENC"),
            ("ZWHD_WNC", "ZWHD_WNC"),
            ("ZWHD_SAC", "ZWHD_SAC"),
            ("ZWHD_ESC", "ZWHD_ESC"),
            ("ZWHD_WSC", "ZWHD_WSC"),
            ("ZWHD_MTN", "ZWHD_MTN"),
            ("ZWHD_PAC", "ZWHD_PAC"),
        ],
    ),
    (
        "STEO ZWHD 10YR",
        [
            ("ZWHD_US_10YR", "ZWHD_US_10YR"),
            ("ZWHD_NEC_10YR", "ZWHD_NEC_10YR"),
            ("ZWHD_MAC_10YR", "ZWHD_MAC_10YR"),
            ("ZWHD_ENC_10YR", "ZWHD_ENC_10YR"),
            ("ZWHD_WNC_10YR", "ZWHD_WNC_10YR"),
            ("ZWHD_SAC_10YR", "ZWHD_SAC_10YR"),
            ("ZWHD_ESC_10YR", "ZWHD_ESC_10YR"),
            ("ZWHD_WSC_10YR", "ZWHD_WSC_10YR"),
            ("ZWHD_MTN_10YR", "ZWHD_MTN_10YR"),
            ("ZWHD_PAC_10YR", "ZWHD_PAC_10YR"),
        ],
    ),
    (
        "STEO ZWCD",
        [
            ("ZWCDPUS", "ZWCDPUS"),
            ("ZWCD_NEC", "ZWCD_NEC"),
            ("ZWCD_MAC", "ZWCD_MAC"),
            ("ZWCD_ENC", "ZWCD_ENC"),
            ("ZWCD_WNC", "ZWCD_WNC"),
            ("ZWCD_SAC", "ZWCD_SAC"),
            ("ZWCD_ESC", "ZWCD_ESC"),
            ("ZWCD_WSC", "ZWCD_WSC"),
            ("ZWCD_MTN", "ZWCD_MTN"),
            ("ZWCD_PAC", "ZWCD_PAC"),
        ],
    ),
    (
        "STEO ZWCD 10YR",
        [
            ("ZWCD_US_10YR", "ZWCD_US_10YR"),
            ("ZWCD_NEC_10YR", "ZWCD_NEC_10YR"),
            ("ZWCD_MAC_10YR", "ZWCD_MAC_10YR"),
            ("ZWCD_ENC_10YR", "ZWCD_ENC_10YR"),
            ("ZWCD_WNC_10YR", "ZWCD_WNC_10YR"),
            ("ZWCD_SAC_10YR", "ZWCD_SAC_10YR"),
            ("ZWCD_ESC_10YR", "ZWCD_ESC_10YR"),
            ("ZWCD_WSC_10YR", "ZWCD_WSC_10YR"),
            ("ZWCD_MTN_10YR", "ZWCD_MTN_10YR"),
            ("ZWCD_PAC_10YR", "ZWCD_PAC_10YR"),
        ],
    ),
]

EIA_COVARIATE_CODES = [
    series_id for _, items in EIA_COVARIATE_GROUPS for series_id, _ in items
]

EIA_COVARIATE_SERIES = {code: f"STEO.{code}.M" for code in EIA_COVARIATE_CODES}

INDICATOR_GROUPS = DIESEL_DEMAND_GROUPS + EIA_COVARIATE_GROUPS
INDICATOR_IDS = [series_id for _, items in INDICATOR_GROUPS for series_id, _ in items]
INDICATOR_META = {
    series_id: {"label": label, "group": group}
    for group, items in INDICATOR_GROUPS
    for series_id, label in items
}

__all__ = [
    "DIESEL_DEMAND_GROUPS",
    "DIESEL_DEMAND_IDS",
    "DIESEL_DEMAND_META",
    "EIA_COVARIATE_GROUPS",
    "EIA_COVARIATE_CODES",
    "EIA_COVARIATE_SERIES",
    "INDICATOR_GROUPS",
    "INDICATOR_IDS",
    "INDICATOR_META",
]
