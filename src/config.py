import os

# ==================== DATA PATHS ====================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_PATH = "../data/raw/Spam_SMS.csv"
DATA_PROCESSED_PATH = "../data/processed/"


# ==================== DATA SPLIT RATIOS ====================
TRAIN_SIZE = 0.6
VAL_SIZE = 0.2
TEST_SIZE = 0.2

# ==================== RANDOM SEED ====================
RANDOM_SEED = 42

# ==================== TEXT PREPROCESSING ====================
REMOVE_STOPWORDS = True
MIN_MESSAGE_LENGTH = 1
