import logging
import re
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# setup logging instead of print
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class DataPreprocessor:
    """
    Handle all data preprocessing steps
    """
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        self.stopwords = set(stopwords.words('english'))
        np.random.seed(random_seed)

    @staticmethod
    def load_data(filepath):
        """
        load and inspect the raw dataset
        :param filepath: (str) path to CSV file
        :return: pd.DataFrame : Loaded dataset
        """
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath, encoding='utf-8')

        # Rename columns for clarity
        if 'Class' in df.columns and 'Message' in df.columns:
            df = df.rename(columns={'Class' : 'label', 'Message' : 'message'})

        # Keep only relevant columns
        df = df[['label', 'message']]

        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Data types:\n{df.dtypes}")
        logger.info(f"Label distribution:\n{df['label'].value_counts()}")
        logger.info(f"Missing values:\n{df.isnull().sum()}")
        logger.info(f"Duplicates: {df.duplicated().sum()}")

        return df
    @staticmethod
    def clean_text(text):
        """
        Clean and normalize the text data
        :param text: (str) raw text data
        :return: str: cleaned text
        """
        if not isinstance(text, str):
            return ""
        # Convert to lowercase
        text = text.lower()

        #Remove URLs
        text = re.sub(r"http\S+|www\s+|https\s+", "", text)

        #Remove email addresses
        text = re.sub(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', '', text)

        # Remove phone numbers
        text = re.sub(r'^(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}$', '', text)

        # Remove special characters and digits (keep spaces)
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Remove extra Whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def tokenize_and_remove_stopwords(self, text, remove_stopwords=False):
        """
        Tokenize and optionally remove stopwords
        :param text: (str) Text to tokenize
        :param remove_stopwords: (bool) whether to remove stopwords
        :return: list of tokens
        """
        tokens = word_tokenize(text)
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]

        return tokens

    def preprocess_data(self, df, remove_stopwords=False):
        """
        complete preprocessing pipeline
        :param df: (pd.DataFrame) raw dataframe
        :param remove_stopwords: (bool) whether to remove stopwords
        :return: pd.DataFrame: Preprocessed dataframe
        """
        logger.info("Starting data preprocessing...")

        # Remove duplicates
        initial_size = len(df)
        df = df.drop_duplicates(subset=['message'])
        logger.info(f"Removed {initial_size - len(df)} duplicate messages")

        # Remove rows with empty messages
        df = df[df['message'].str.strip() != '']
        logger.info(f"Dataset shape after removing empty messages: {df.shape}")

        # Clean text
        logger.info("Cleaning text data...")
        df['message'] = df['message'].apply(self.clean_text)

        # Remove any rows that became empty after cleaning
        df = df[df['message'].str.strip() != '']
        logger.info(f"Dataset shape after cleaning: {df.shape}")

        # Tokenize (optional stopword removal)
        logger.info("Tokenizing text...")
        df['tokens'] = df['message'].apply(
            lambda x: self.tokenize_and_remove_stopwords(x, remove_stopwords)
        )

        # Add features
        df['message_length'] = df['message'].str.len()
        df['word_count'] = df['message'].str.split().str.len()
        df['avg_word_length'] = df['message_length'] / df['word_count'].replace(0, 1)

        # Convert label to binary
        df['label'] = (df['label'] == 'spam').astype(int)

        logger.info(f"Preprocessing complete. Final shape: {df.shape}")
        logger.info(f"Label distribution:\n{df['label'].value_counts()}")

        return df

    def split_data(self, df, train_size=0.6, val_size=0.2, test_size=0.2):
        """
        split data into train, validation and test sets
        :param df:(pd.DataFrame) preprocessed dataframe
        :param train_size: (float) proportion for training set
        :param val_size: (float) proportion for validation set
        :param test_size: (float) proportion for test set
        :return: tuple: (train_df, val_df, test_df)
        """
        # Verify sizes sum to 1
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
            "Train, val, test sizes must sum to 1"

        logger.info("Splitting data with stratification...")

        # First split: train + rest
        train_df, temp_df = train_test_split(
            df,
            train_size=train_size,
            stratify=df['label'],
            random_state=self.random_seed
        )

        # Second split: val and test
        val_test_ratio = val_size / (val_size + test_size)
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_test_ratio,
            stratify=temp_df['label'],
            random_state=self.random_seed
        )

        logger.info(f"Train set size: {len(train_df)} ({len(train_df) / len(df) * 100:.1f}%)")
        logger.info(f"Train label distribution:\n{train_df['label'].value_counts()}")
        logger.info(f"\nValidation set size: {len(val_df)} ({len(val_df) / len(df) * 100:.1f}%)")
        logger.info(f"Validation label distribution:\n{val_df['label'].value_counts()}")
        logger.info(f"\nTest set size: {len(test_df)} ({len(test_df) / len(df) * 100:.1f}%)")
        logger.info(f"Test label distribution:\n{test_df['label'].value_counts()}")

        return train_df, val_df, test_df
    @staticmethod
    def save_processed_data(train_df, val_df, test_df, output_dir):
        """
        Save processed datasets to CSV files
        :param train_df: (pd.DataFrame) preprocessed dataframe
        :param val_df: (pd.DataFrame) preprocessed dataframe
        :param test_df: (pd.DataFrame) preprocessed dataframe
        :param output_dir: (str) output directory path
        :return: None
        """
        os.makedirs(output_dir, exist_ok=True)

        train_df.to_csv(f"{output_dir}/train_data.csv", index=False)
        val_df.to_csv(f"{output_dir}/val_data.csv", index=False)
        test_df.to_csv(f"{output_dir}/test_data.csv", index=False)

        logger.info(f"Processed data saved to {output_dir}")

