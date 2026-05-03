TIMESERIES_CURVES = [
    # --- MCP (DA Price) ---
    "DE Price Spot EUR/MWh EPEX 15min Actual",
    "DE Price Spot EUR/MWh EPEX H Actual",

    # --- Cross-Border and Network ---
    "DE Exchange Day-Ahead Schedule Net Import MWh/h H Actual",

    # --- Cross-Border NTC (REMIT) ---
    "AT>DE Exchange Net Transfer Capacity MW 15min REMIT",
    "BE>DE Exchange Net Transfer Capacity MW 15min REMIT",
    "CH>DE Exchange Net Transfer Capacity MW 15min REMIT",
    "CZ>DE Exchange Net Transfer Capacity MW 15min REMIT",
    "DK1>DE Exchange Net Transfer Capacity MW 15min REMIT",
    "DK2>DE Exchange Net Transfer Capacity MW 15min REMIT",
    "FR>DE Exchange Net Transfer Capacity MW 15min REMIT",
    "NL>DE Exchange Net Transfer Capacity MW 15min REMIT",
    "NO2>DE Exchange Net Transfer Capacity MW 15min REMIT",
    "SE4>DE Exchange Net Transfer Capacity MW 15min REMIT",
    "DE>AT Exchange Net Transfer Capacity MW 15min REMIT",
    "DE>BE Exchange Net Transfer Capacity MW 15min REMIT",
    "DE>CH Exchange Net Transfer Capacity MW 15min REMIT",
    "DE>CZ Exchange Net Transfer Capacity MW 15min REMIT",
    "DE>DK1 Exchange Net Transfer Capacity MW 15min REMIT",
    "DE>DK2 Exchange Net Transfer Capacity MW 15min REMIT",
    "DE>FR Exchange Net Transfer Capacity MW 15min REMIT",
    "DE>NL Exchange Net Transfer Capacity MW 15min REMIT",
    "DE>NO2 Exchange Net Transfer Capacity MW 15min REMIT",
    "DE>SE4 Exchange Net Transfer Capacity MW 15min REMIT",

    # --- Actual Values for Lag Variables
    "DE Wind Power Production MWh/h 15min Actual",
    "DE Wind Power Production Offshore MWh/h 15min Actual",
    "DE Wind Power Production Onshore MWh/h 15min Actual",
    "DE Solar Photovoltaic Production MWh/h 15min Actual",
    "DE Residual Load MWh/h 15min Actual",
    "DE Consumption MWh/h 15min Actual",

    # --- Intraday Prices (for DA-ID strategy) ---
    "DE Price Intraday VWAP EUR/MWh EPEX 15min Actual",
    "DE Price Intraday VWAP EUR/MWh EPEX 30min Actual",
    "DE Price Intraday VWAP EUR/MWh EPEX H Actual",
    "DE Price Intraday VWAP ID1 EUR/MWh EPEX 15min Actual",
    "DE Price Intraday VWAP ID1 EUR/MWh EPEX 30min Actual",
    "DE Price Intraday VWAP ID1 EUR/MWh EPEX H Actual",
    "DE Price Intraday VWAP ID3 EUR/MWh EPEX 15min Actual",
    "DE Price Intraday VWAP ID3 EUR/MWh EPEX 30min Actual",
    "DE Price Intraday VWAP ID3 EUR/MWh EPEX H Actual",


    # --- Imbalance Price and Volume (for DA-IMB strategy) ---
    "DE Price Imbalance Single EUR/MWh 15min Actual",
    "DE Volume Imbalance Net MWh 15min Actual"
]


SCENARIO_TIMESERIES_CURVES = [
    # --- Demand / Load ---
    "DE Residual Load MWh/h 15min Climate",
    "DE Consumption MWh/h 15min Climate",

    # --- Renewable Generation ---
    "DE Wind Power Production MWh/h 15min Climate",
    "DE Wind Power Production Offshore MWh/h 15min Climate",
    "DE Wind Power Production Onshore MWh/h 15min Climate",
    "DE Solar Photovoltaic Production MWh/h 15min Climate",
]

OHLC_CURVES = [
    "NL Futures Natural Gas EUR/MWh ICE-TTF OHLC",
    "DK Futures Natural Gas EUR/MWh EEX-ETF OHLC",
    "DE Futures Natural Gas EUR/MWh EEX-THE OHLC",
    "Futures EUA EUR/t ICE OHLC",
    "Futures Coal API-2 USD/t ICE OHLC",
    "Futures Gasoil USD/t ICE OHLC",
    "Futures Crude Oil Brent USD/bbl ICE OHLC",
    "DE Futures Power Base EUR/MWh EEX OHLC",
    "DE Futures Power Peak EUR/MWh EEX OHLC",
]


INSTANCE_CURVES = [
    # --- MCP (DA Price) ---
    "DE Price Spot EUR/MWh 15min Forecast",
    "DE Price Spot EUR/MWh H Forecast",
    "DE Price Spot Ensemble EUR/MWh 15min Forecast",
    "DE Price Spot Ensemble EUR/MWh H Forecast",

    # --- Demand / Load ---
    "DE Residual Load MWh/h 15min Forecast",
    "DE Consumption MWh/h 15min Forecast",
    "DE Renewable Power Consumption MWh/h 15min Forecast",
    "DE Renewable Power Consumption % 15min Forecast",
    "DE Low-carbon Power Consumption % 15min Forecast",
    "DE Low-carbon Power Consumption MWh/h 15min Forecast",

    # --- Renewable Generation ---
    "DE Wind Power Production MWh/h 15min Forecast",
    "DE Wind Power Production Offshore MWh/h 15min Forecast",
    "DE Wind Power Production Onshore MWh/h 15min Forecast",
    "DE Solar Photovoltaic Production MWh/h 15min Forecast",

    # --- France Nuclear ---
    "FR Nuclear Production MWh/h 15min Forecast",

    # --- Cross-Border and Network (Schedules) ---
    "DE Exchange Day-Ahead Schedule Net Export MWh/h H Forecast",
    "AT>DE Exchange Day-Ahead Schedule MWh/h H Forecast",
    "BE>DE Exchange Day-Ahead Schedule MWh/h H Forecast",
    "DE>AT Exchange Day-Ahead Schedule MWh/h H Forecast",
    "DE>BE Exchange Day-Ahead Schedule MWh/h H Forecast",
    "CH>DE Exchange Day-Ahead Schedule MWh/h H Forecast",
    "DE>CH Exchange Day-Ahead Schedule MWh/h H Forecast",
    "DE>FR Exchange Day-Ahead Schedule MWh/h H Forecast",
    "DE>NL Exchange Day-Ahead Schedule MWh/h H Forecast",
    "DE>PL Exchange Day-Ahead Schedule MWh/h H Forecast",
    "FR>DE Exchange Day-Ahead Schedule MWh/h H Forecast",
    "NL>DE Exchange Day-Ahead Schedule MWh/h H Forecast",
    "PL>DE Exchange Day-Ahead Schedule MWh/h H Forecast",
    "CZ>DE Exchange Day-Ahead Schedule MWh/h H Forecast",
    "DE>CZ Exchange Day-Ahead Schedule MWh/h H Forecast",
    "DE>DK1 Exchange Day-Ahead Schedule MWh/h H Forecast",
    "DE>DK2 Exchange Day-Ahead Schedule MWh/h H Forecast",
    "DE>NO2 Exchange Day-Ahead Schedule MWh/h H Forecast",
    "DE>SE4 Exchange Day-Ahead Schedule MWh/h H Forecast",
    "DK1>DE Exchange Day-Ahead Schedule MWh/h H Forecast",
    "DK2>DE Exchange Day-Ahead Schedule MWh/h H Forecast",
    "NO2>DE Exchange Day-Ahead Schedule MWh/h H Forecast",
    "SE4>DE Exchange Day-Ahead Schedule MWh/h H Forecast",

    # --- Neighbor Spot Price Forecasts ---
    "AT Price Spot EUR/MWh 15min Forecast",
    "BE Price Spot EUR/MWh 15min Forecast",
    "CH Price Spot EUR/MWh 15min Forecast",
    "CZ Price Spot EUR/MWh 15min Forecast",
    "DK2 Price Spot EUR/MWh 15min Forecast",
    "FR Price Spot EUR/MWh 15min Forecast",
    "NL Price Spot EUR/MWh 15min Forecast",
    "PL Price Spot EUR/MWh 15min Forecast",
    "AT Price Spot EUR/MWh H Forecast",
    "BE Price Spot EUR/MWh H Forecast",
    "CH Price Spot EUR/MWh H Forecast",
    "CZ Price Spot EUR/MWh H Forecast",
    "DK2 Price Spot EUR/MWh H Forecast",
    "FR Price Spot EUR/MWh H Forecast",
    "NL Price Spot EUR/MWh H Forecast",
    "PL Price Spot EUR/MWh H Forecast",

    # --- Neighbor Residual Load Forecasts ---
    "AT Residual Load MWh/h 15min Forecast",
    "BE Residual Load MWh/h 15min Forecast",
    "CH Residual Load MWh/h 15min Forecast",
    "CZ Residual Load MWh/h 15min Forecast",
    "DK Residual Load MWh/h 15min Forecast",
    "FR Residual Load MWh/h 15min Forecast",
    "NL Residual Load MWh/h 15min Forecast",
    "PL Residual Load MWh/h 15min Forecast",

    # --- Neighbor Temperature Forecasts ---
    "AT Consumption Temperature °C 15min Forecast",
    "BE Consumption Temperature °C 15min Forecast",
    "DE Consumption Temperature °C 15min Forecast",
    "CH Consumption Temperature °C 15min Forecast",
    "CZ Consumption Temperature °C 15min Forecast",
    "DK Consumption Temperature °C 15min Forecast",
    "FR Consumption Temperature °C 15min Forecast",
    "NL Consumption Temperature °C 15min Forecast",
    "PL Consumption Temperature °C 15min Forecast",
]