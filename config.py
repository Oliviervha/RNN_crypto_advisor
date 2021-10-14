"""
Configuration

"""

class Configuration:

    NO_RUNS = 1

    # data parameters
    PATHTOFILE = './data/VeThor_USD.xlsx' # make sure data is ascending by date
    PRICE_COL   = 'Close'
    DATE_COL    = 't'
    VOL_COL     = 'Volume'
    OI_COL      = 'OI'
    CVD_COL     = 'CVD'
    DELTA_COL   = 'Delta'
    MARKET_CAP_COL = 'market_cap'
    BTC_CLOSE   = 'Close_BTC'
    BTC_VOL     = 'Vol_BTC'

    FEATURES = [PRICE_COL, VOL_COL, BTC_CLOSE, BTC_VOL]

    # Model parameters
    OUTPUT_SIZE = 1
    INPUT_SIZE  = 5
    N_FEATURES = len(FEATURES)
    EPOCHS = 2000
    LR = 0.01

    MIN_PRICE_INCREASE = 0.001 # 0.1 = 10%

    TRAIN_SIZE  = 0.7
    VAL_SIZE    = 0.15
    TEST_SIZE   = 0.15

    MIN_PROBA = 0.9 #>
    RANDOM_SAMPLE = 100

    SAVEMODEL = False
