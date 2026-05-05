"""
Module 1-2: Data Ingestion, Parsing, Feature Engineering
Handles CERT dataset ingestion and feature extraction
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# def load_cert_data(data_dir='insider_threat_detection/data'):
#     logon = pd.read_csv(f"{data_dir}/logon.csv").head(5000)
#     file = pd.read_csv(f"{data_dir}/file.csv").head(5000)
#     email = pd.read_csv(f"{data_dir}/email.csv").head(5000)
#     insiders = pd.read_csv(f"{data_dir}/insiders.csv")

#     return logon, file, email, insiders
def load_cert_data():
    logon = pd.read_csv("data/logon.csv").sample(20000)
    file = pd.read_csv("data/file.csv").sample(20000)
    email = pd.read_csv("data/email.csv").sample(20000)
    insiders = pd.read_csv("data/insiders.csv")

    return logon, file, email, insiders

class DataPipeline:
    """Handles data ingestion, cleaning, and feature engineering"""
    
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = data_dir
        self.logs = {}
        self.features_df = None
        self.user_profiles = {}
        
    def generate_synthetic_cert_data(self, n_users: int = 1000, n_days: int = 500, 
                                      malicious_ratio: float = 0.05):
        """
        Generate synthetic CERT-like dataset
        Real CERT dataset would be loaded here instead
        """
        print(f"[*] Generating synthetic CERT dataset: {n_users} users, {n_days} days")
        
        np.random.seed(42)
        start_date = datetime(2023, 1, 1)
        
        # Generate malicious users
        n_malicious = int(n_users * malicious_ratio)
        malicious_users = [f"user_{i:04d}" for i in range(n_malicious)]
        all_users = [f"user_{i:04d}" for i in range(n_users)]
        
        # Logon logs
        logon_logs = []
        for user in all_users:
            is_malicious = user in malicious_users
            n_events = np.random.randint(200, 400) if not is_malicious else np.random.randint(250, 500)
            
            for _ in range(n_events):
                timestamp = start_date + timedelta(
                    days=np.random.randint(0, n_days),
                    hours=np.random.randint(0, 24),
                    minutes=np.random.randint(0, 60)
                )
                
                # Malicious users have more off-hours access
                if is_malicious and np.random.rand() < 0.3:
                    timestamp = timestamp.replace(hour=np.random.randint(22, 24) or np.random.randint(0, 6))
                
                success = not (is_malicious and np.random.rand() < 0.1)
                
                logon_logs.append({
                    'date': timestamp.date(),
                    'user': user,
                    'event_type': 'logon',
                    'success': success,
                    'timestamp': timestamp
                })
        
        self.logs['logon'] = pd.DataFrame(logon_logs)
        
        # File access logs
        file_logs = []
        for user in all_users:
            is_malicious = user in malicious_users
            n_events = np.random.randint(50, 150) if not is_malicious else np.random.randint(100, 300)
            
            for _ in range(n_events):
                timestamp = start_date + timedelta(
                    days=np.random.randint(0, n_days),
                    hours=np.random.randint(0, 24),
                    minutes=np.random.randint(0, 60)
                )
                
                action = np.random.choice(['read', 'write', 'copy', 'delete'])
                
                file_logs.append({
                    'date': timestamp.date(),
                    'user': user,
                    'event_type': 'file',
                    'action': action,
                    'timestamp': timestamp
                })
        
        self.logs['file'] = pd.DataFrame(file_logs)
        
        # Email logs
        email_logs = []
        for user in all_users:
            is_malicious = user in malicious_users
            n_events = np.random.randint(20, 60) if not is_malicious else np.random.randint(30, 100)
            
            for _ in range(n_events):
                timestamp = start_date + timedelta(
                    days=np.random.randint(0, n_days),
                    hours=np.random.randint(0, 24),
                    minutes=np.random.randint(0, 60)
                )
                
                n_recipients = np.random.randint(1, 10)
                external_recipients = np.random.randint(0, n_recipients + 1)
                
                email_logs.append({
                    'date': timestamp.date(),
                    'user': user,
                    'event_type': 'email',
                    'recipients': n_recipients,
                    'external_recipients': external_recipients,
                    'timestamp': timestamp
                })
        
        self.logs['email'] = pd.DataFrame(email_logs)
        
        # HTTP logs
        http_logs = []
        suspicious_domains = ['malware.com', 'exfil.io', 'c2-server.ru', 'data-sell.dark']
        
        for user in all_users:
            is_malicious = user in malicious_users
            n_events = np.random.randint(100, 300) if not is_malicious else np.random.randint(150, 500)
            
            for _ in range(n_events):
                timestamp = start_date + timedelta(
                    days=np.random.randint(0, n_days),
                    hours=np.random.randint(0, 24),
                    minutes=np.random.randint(0, 60)
                )
                
                domain = 'normal.com'
                if is_malicious and np.random.rand() < 0.15:
                    domain = np.random.choice(suspicious_domains)
                
                http_logs.append({
                    'date': timestamp.date(),
                    'user': user,
                    'event_type': 'http',
                    'domain': domain,
                    'timestamp': timestamp
                })
        
        self.logs['http'] = pd.DataFrame(http_logs)
        
        # USB logs
        usb_logs = []
        for user in all_users:
            is_malicious = user in malicious_users
            n_events = np.random.randint(5, 20) if not is_malicious else np.random.randint(15, 50)
            
            for _ in range(n_events):
                timestamp = start_date + timedelta(
                    days=np.random.randint(0, n_days),
                    hours=np.random.randint(0, 24),
                    minutes=np.random.randint(0, 60)
                )
                
                usb_logs.append({
                    'date': timestamp.date(),
                    'user': user,
                    'event_type': 'usb',
                    'action': np.random.choice(['insert', 'copy', 'eject']),
                    'timestamp': timestamp
                })
        
        self.logs['usb'] = pd.DataFrame(usb_logs)
        
        print(f"[+] Dataset generated successfully")
        print(f"    Malicious users: {n_malicious}")
        print(f"    Total users: {n_users}")
        
        return malicious_users, all_users
    
    # def extract_features(self) -> pd.DataFrame:
    #     """Extract daily user features from logs"""
    #     print("[*] Extracting features...")
        
    #     all_dates = []
    #     all_users = set()
        
    #     for log_type, df in self.logs.items():
    #         all_dates.extend(df['date'].unique())
    #         all_users.update(df['user'].unique())
        
    #     date_range = pd.date_range(
    #         start=min(all_dates),
    #         end=max(all_dates),
    #         freq='D'
    #     )
        
    #     features_list = []
        
    #     for user in sorted(all_users):
    #         for date in date_range:
    #             features = {'user': user, 'date': date.date()}
                
    #             # Logon features
    #             logon_df = self.logs['logon']
    #             user_logon = logon_df[
    #                 (logon_df['user'] == user) & (logon_df['date'] == date.date())
    #             ]
    #             features['logon_count'] = len(user_logon)
    #             features['failed_logins'] = (user_logon['success'] == False).sum()
    #             features['off_hours_access'] = sum(
    #                 1 for ts in user_logon['timestamp'] if ts.hour < 6 or ts.hour > 22
    #             )
                
    #             # File features
    #             file_df = self.logs['file']
    #             user_file = file_df[(file_df['user'] == user) & (file_df['date'] == date.date())]
    #             features['files_accessed'] = len(user_file)
    #             features['files_copied'] = (user_file['action'] == 'copy').sum()
    #             features['files_deleted'] = (user_file['action'] == 'delete').sum()
                
    #             # Email features
    #             email_df = self.logs['email']
    #             user_email = email_df[(email_df['user'] == user) & (email_df['date'] == date.date())]
    #             features['emails_sent'] = len(user_email)
    #             features['external_recipients'] = user_email['external_recipients'].sum()
                
    #             # HTTP features
    #             http_df = self.logs['http']
    #             user_http = http_df[(http_df['user'] == user) & (http_df['date'] == date.date())]
    #             features['websites_visited'] = len(user_http)
    #             features['suspicious_domains'] = (
    #                 user_http['domain'].isin(['malware.com', 'exfil.io', 'c2-server.ru', 'data-sell.dark']).sum()
    #             )
                
    #             # USB features
    #             usb_df = self.logs['usb']
    #             user_usb = usb_df[(usb_df['user'] == user) & (usb_df['date'] == date.date())]
    #             features['usb_usage'] = len(user_usb)
                
    #             features_list.append(features)
        
    #     self.features_df = pd.DataFrame(features_list)
        
    #     # Normalize dates to datetime
    #     self.features_df['date'] = pd.to_datetime(self.features_df['date'])
        
    #     print(f"[+] Features extracted: {len(self.features_df)} records")
    #     print(f"    Features per day: {len(self.features_df.columns) - 2}")
        
    #     return self.features_df
    def extract_features(self) -> pd.DataFrame:
        print("USING REAL CERT DATASET")

        logon, file, email, insiders = load_cert_data()

        # Normalize column names
        logon.columns = logon.columns.str.lower()
        file.columns = file.columns.str.lower()
        email.columns = email.columns.str.lower()
        insiders.columns = insiders.columns.str.lower()

        # Convert timestamp → datetime (IMPORTANT: keep full timestamp first)
        logon['datetime'] = pd.to_datetime(logon['date'])
        file['datetime'] = pd.to_datetime(file['date'])
        email['datetime'] = pd.to_datetime(email['date'])

        # Extract date + hour
        logon['date'] = logon['datetime'].dt.date
        file['date'] = file['datetime'].dt.date
        email['date'] = email['datetime'].dt.date

        logon['hour'] = logon['datetime'].dt.hour

        # =========================
        # 🔹 LOGON FEATURES
        # =========================
        logon_feat = logon.groupby(['user', 'date']).agg(
            logon_count=('user', 'count')
        ).reset_index()

        # Off-hours activity (before 6 AM or after 10 PM)
        off_hours = logon[(logon['hour'] < 6) | (logon['hour'] > 22)]
        off_hours_feat = off_hours.groupby(['user', 'date']).size().reset_index(name='off_hours_logons')

        logon_feat = logon_feat.merge(off_hours_feat, on=['user', 'date'], how='left')
        logon_feat['off_hours_logons'] = logon_feat['off_hours_logons'].fillna(0)

        # =========================
        # 🔹 FILE FEATURES
        # =========================
        file_feat = file.groupby(['user', 'date']).agg(
            files_accessed=('user', 'count')
        ).reset_index()

        # File actions (if column exists)
        if 'activity' in file.columns:
            file['activity'] = file['activity'].str.lower()

            copied = file[file['activity'] == 'copy'].groupby(['user', 'date']).size().reset_index(name='files_copied')
            deleted = file[file['activity'] == 'delete'].groupby(['user', 'date']).size().reset_index(name='files_deleted')

            file_feat = file_feat.merge(copied, on=['user', 'date'], how='left')
            file_feat = file_feat.merge(deleted, on=['user', 'date'], how='left')

        file_feat = file_feat.fillna(0)

        # =========================
        # 🔹 EMAIL FEATURES
        # =========================
        email_feat = email.groupby(['user', 'date']).agg(
            emails_sent=('user', 'count')
        ).reset_index()

        # External emails (if column exists)
        if 'to' in email.columns:
            email['external'] = email['to'].str.contains('@', na=False).astype(int)
            external_feat = email.groupby(['user', 'date'])['external'].sum().reset_index(name='external_emails')

            email_feat = email_feat.merge(external_feat, on=['user', 'date'], how='left')

        email_feat = email_feat.fillna(0)

        # =========================
        # 🔹 MERGE ALL FEATURES
        # =========================
        df = logon_feat.merge(file_feat, on=['user', 'date'], how='outer')
        df = df.merge(email_feat, on=['user', 'date'], how='outer')

        df = df.fillna(0)

        # =========================
        # 🔹 LABEL (INSIDER THREAT)
        # =========================
        df['label'] = df['user'].isin(insiders['user']).astype(int)

        self.features_df = df

        print(f"[+] Features extracted: {len(df)} records")
        print(f"[+] Unique users: {df['user'].nunique()}")

        return self.features_df
    
    def normalize_features(self) -> pd.DataFrame:
        """Normalize numerical features"""
        print("[*] Normalizing features...")
        
        feature_cols = [col for col in self.features_df.columns if col not in ['user', 'date']]
        
        # Fill NaN with 0
        self.features_df[feature_cols] = self.features_df[feature_cols].fillna(0)
        
        # Normalize to 0-1 range per feature
        for col in feature_cols:
            max_val = self.features_df[col].max()
            if max_val > 0:
                self.features_df[col + '_norm'] = self.features_df[col] / max_val
        
        print(f"[+] Normalization complete")
        
        return self.features_df
    
    def get_feature_matrix(self) -> Tuple[np.ndarray, List[str], List[str]]:
        """Get feature matrix for model training"""
        feature_cols = [col for col in self.features_df.columns 
                       if col not in ['user', 'date'] and not col.endswith('_norm')]
        
        X = self.features_df[feature_cols].values
        users = self.features_df['user'].values
        dates = self.features_df['date'].values
        
        return X, users, dates, feature_cols


if __name__ == "__main__":
    # Test data pipeline
    pipeline = DataPipeline()
    # malicious_users, all_users = pipeline.generate_synthetic_cert_data(
    #     n_users=200,
    #     n_days=500,
    #     malicious_ratio=0.05
    # )
    
    features_df = pipeline.extract_features()
    pipeline.normalize_features()
    
    X, users, dates, feature_cols = pipeline.get_feature_matrix()
    
    print(f"\n[+] Data pipeline complete!")
    print(f"    Feature matrix shape: {X.shape}")
    # print(f"    Malicious users: {malicious_users[:5]}...")
