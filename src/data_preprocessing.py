"""
Data preprocessing pipeline for:
  - Hotel Booking Demand  (primary experiment)
  - IBM Telco Customer Churn (generalization)

Returns train / val / test splits as numpy arrays + PyTorch DataLoaders.
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.impute           import SimpleImputer
from imblearn.over_sampling   import SMOTE

import torch
from torch.utils.data import Dataset, DataLoader

from src.config import (
    HOTEL_CSV, TELCO_CSV, BANK_CSV, DATATHON_DIR, SEED,
    TEST_SIZE, VAL_SIZE, CV_FOLDS, DEVICE
)


# ─────────────────────────────────────────────────────────────────────────────
# PYTORCH DATASET
# ─────────────────────────────────────────────────────────────────────────────

class ChurnDataset(Dataset):
    """
    Wraps numpy arrays into a PyTorch Dataset.
    Returns (num_features, cat_features, label) per sample.
    """

    def __init__(self, X_num: np.ndarray, X_cat: np.ndarray, y: np.ndarray):
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.y     = torch.tensor(y,     dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_num[idx], self.X_cat[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# HOTEL BOOKING DEMAND
# ─────────────────────────────────────────────────────────────────────────────

class HotelDataProcessor:
    """
    Full preprocessing pipeline for Hotel Booking Demand dataset.
    Target: is_canceled  (1 = churn / cancellation)
    """

    CATEGORICAL_COLS = [
        "hotel", "arrival_date_month", "meal", "market_segment",
        "distribution_channel", "reserved_room_type", "assigned_room_type",
        "deposit_type", "customer_type",
    ]

    NUMERICAL_COLS = [
        "lead_time", "arrival_date_year", "arrival_date_week_number",
        "arrival_date_day_of_month", "stays_in_weekend_nights",
        "stays_in_week_nights", "adults", "children", "babies",
        "is_repeated_guest", "previous_cancellations",
        "previous_bookings_not_canceled", "booking_changes",
        "days_in_waiting_list", "adr", "required_car_parking_spaces",
        "total_of_special_requests",
    ]

    DROP_COLS = [
        "reservation_status",        # data leakage
        "reservation_status_date",   # data leakage
        "agent", "company",          # too many missing / high cardinality
        "country",                   # very high cardinality
    ]

    TARGET_COL = "is_canceled"

    def __init__(self):
        self.label_encoders = {}
        self.scaler         = StandardScaler()
        self.imputer_num    = SimpleImputer(strategy="median")
        self.imputer_cat    = SimpleImputer(strategy="most_frequent")
        self.cat_dims_      = []      # cardinalities after encoding
        self.fitted_        = False

    # ── feature engineering ──────────────────────────────────────────────────

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # ── Existing features ────────────────────────────────────────────────

        # Total nights
        df["total_nights"] = (
            df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
        )

        # Room mismatch (reserved ≠ assigned)
        df["room_mismatch"] = (
            df["reserved_room_type"] != df["assigned_room_type"]
        ).astype(int)

        # Revenue proxy
        df["revenue_proxy"] = df["adr"] * (
            df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
        )
        df["revenue_proxy"] = df["revenue_proxy"].clip(lower=0)

        # Month as cyclic
        month_map = {
            "January":1,"February":2,"March":3,"April":4,
            "May":5,"June":6,"July":7,"August":8,
            "September":9,"October":10,"November":11,"December":12,
        }
        df["month_num"] = df["arrival_date_month"].map(month_map).fillna(0)
        df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)

        # Week of year cyclic
        df["week_sin"] = np.sin(2 * np.pi * df["arrival_date_week_number"] / 52)
        df["week_cos"] = np.cos(2 * np.pi * df["arrival_date_week_number"] / 52)

        # Is weekend booking
        df["has_weekend_nights"] = (df["stays_in_weekend_nights"] > 0).astype(int)

        # Booking behaviour
        df["lead_per_night"] = df["lead_time"] / (df["total_nights"] + 1)

        # ── New domain-specific features ─────────────────────────────────────

        # 1. Cancellation risk ratio — strongest behavioural signal
        #    = fraction of past bookings that were cancelled
        #    (high ratio = chronic canceller)
        df["cancellation_risk_ratio"] = (
            df["previous_cancellations"] /
            (df["previous_cancellations"] + df["previous_bookings_not_canceled"] + 1)
        )

        # 2. Guest party size — total occupants
        df["party_size"] = df["adults"] + df["children"] + df["babies"]

        # 3. ADR per person — price per occupant (luxury proxy)
        df["adr_per_person"] = df["adr"] / (df["party_size"] + 1)

        # 4. Family booking indicator
        #    Families (children/babies) tend to commit more firmly
        df["is_family"] = (
            (df["children"] > 0) | (df["babies"] > 0)
        ).astype(int)

        # 5. Special-request engagement ratio
        #    More requests relative to party size = more invested guest
        df["engagement_ratio"] = (
            df["total_of_special_requests"] / (df["party_size"] + 1)
        )

        # 6. Weekend proportion of stay (leisure vs business indicator)
        df["weekend_proportion"] = (
            df["stays_in_weekend_nights"] / (df["total_nights"] + 1)
        )

        # 7. Booking instability — changes per unit lead time
        #    Frequent changes early in the booking lifecycle = unstable intent
        df["booking_instability"] = (
            df["booking_changes"] / (df["lead_time"] + 1)
        )

        # 8. Loyalty score — repeat guest who has NOT cancelled before
        #    Strong negative churn signal
        df["loyalty_score"] = (
            df["is_repeated_guest"] *
            df["previous_bookings_not_canceled"] /
            (df["previous_cancellations"] + 1)
        ).clip(upper=10)

        # 9. High-risk booking flag
        #    Long lead + no deposit = highest cancellation risk combination
        #    (deposit_type is still raw string here, encoded later)
        df["long_lead_no_commit"] = (
            (df["lead_time"] > 90) &
            (df["deposit_type"] == "No Deposit")
        ).astype(int)

        # 10. Revenue per night — normalised booking value
        df["revenue_per_night"] = df["revenue_proxy"] / (df["total_nights"] + 1)

        new_features = [
            "cancellation_risk_ratio", "party_size", "adr_per_person",
            "is_family", "engagement_ratio", "weekend_proportion",
            "booking_instability", "loyalty_score",
            "long_lead_no_commit", "revenue_per_night",
        ]

        self.NUMERICAL_COLS = list(set(self.NUMERICAL_COLS + [
            "total_nights", "room_mismatch", "revenue_proxy",
            "month_sin", "month_cos", "week_sin", "week_cos",
            "has_weekend_nights", "lead_per_night",
        ] + new_features))

        return df

    # ── main preprocessing ───────────────────────────────────────────────────

    def load_and_clean(self, path: str = None) -> pd.DataFrame:
        path = path or HOTEL_CSV
        df   = pd.read_csv(path)

        # Replace string NULL in agent/company
        df["agent"]   = df["agent"].replace("NULL",   np.nan)
        df["company"] = df["company"].replace("NULL", np.nan)

        # Drop leakage & high-cardinality
        df = df.drop(columns=[c for c in self.DROP_COLS if c in df.columns])

        # Remove impossible ADR rows
        df = df[df["adr"] >= 0].reset_index(drop=True)

        # Feature engineering
        df = self._engineer_features(df)

        return df

    def fit_transform(self, df: pd.DataFrame):
        """Fit encoders/scalers and return (X_num, X_cat, y)."""
        y = df[self.TARGET_COL].values.astype(int)

        # Ensure columns exist after engineering
        num_cols = [c for c in self.NUMERICAL_COLS if c in df.columns]
        cat_cols = [c for c in self.CATEGORICAL_COLS if c in df.columns]

        # Numerical
        X_num = df[num_cols].copy().astype(float)
        X_num = pd.DataFrame(
            self.imputer_num.fit_transform(X_num), columns=num_cols
        )
        X_num = self.scaler.fit_transform(X_num)

        # Categorical
        X_cat_df = df[cat_cols].copy().astype(str).fillna("UNK")
        X_cat_df = pd.DataFrame(
            self.imputer_cat.fit_transform(X_cat_df), columns=cat_cols
        )
        self.cat_dims_ = []
        for c in cat_cols:
            le = LabelEncoder()
            X_cat_df[c] = le.fit_transform(X_cat_df[c])
            self.label_encoders[c] = le
            self.cat_dims_.append(len(le.classes_))

        X_cat = X_cat_df.values.astype(int)
        self.fitted_ = True
        self._num_cols = num_cols
        self._cat_cols = cat_cols

        return X_num, X_cat, y

    def transform(self, df: pd.DataFrame):
        """Transform unseen data using fitted encoders."""
        assert self.fitted_, "Call fit_transform first."
        y = df[self.TARGET_COL].values.astype(int)

        num_cols = self._num_cols
        cat_cols = self._cat_cols

        X_num = df[num_cols].copy().astype(float)
        X_num = pd.DataFrame(
            self.imputer_num.transform(X_num), columns=num_cols
        )
        X_num = self.scaler.transform(X_num)

        X_cat_df = df[cat_cols].copy().astype(str).fillna("UNK")
        X_cat_df = pd.DataFrame(
            self.imputer_cat.transform(X_cat_df), columns=cat_cols
        )
        for c in cat_cols:
            le = self.label_encoders[c]
            # Vectorized unseen-label handling via lookup dict
            lookup = {cls: i for i, cls in enumerate(le.classes_)}
            X_cat_df[c] = X_cat_df[c].map(lambda x: lookup.get(x, 0)).astype(int)

        X_cat = X_cat_df.values.astype(int)
        return X_num, X_cat, y


# ─────────────────────────────────────────────────────────────────────────────
# IBM TELCO CHURN
# ─────────────────────────────────────────────────────────────────────────────

class TelcoDataProcessor:
    """
    Full preprocessing pipeline for IBM Telco Churn dataset.
    Target: ChurnLabel  (Yes=1, No=0)
    """

    DROP_COLS = [
        "CustomerID", "Country", "State", "City", "ZipCode",
        "Latitude", "Longitude", "Population",
        "CustomerStatus",             # leakage (post-outcome status)
        "ChurnScore", "ChurnCategory","ChurnReason", "CLTV",
        "SatisfactionScore",          # leakage: post-hoc survey score, ~monotone with churn
        "Under30",                    # redundant with Age
    ]

    BINARY_COLS = [
        "Gender", "SeniorCitizen", "Married", "Dependents",
        "ReferredaFriend", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtectionPlan", "PremiumTechSupport",
        "StreamingTV", "StreamingMovies", "StreamingMusic",
        "UnlimitedData", "PaperlessBilling",
    ]

    CATEGORICAL_COLS = ["Offer", "InternetType", "Contract", "PaymentMethod", "Quarter"]

    NUMERICAL_COLS = [
        "Age", "NumberofDependents", "Number_of_Referrals",
        "TenureinMonths", "AvgMonthlyLongDistanceCharges",
        "AvgMonthlyGBDownload", "MonthlyCharge", "TotalCharges",
        "TotalRefunds", "TotalExtraDataCharges",
        "TotalLongDistanceCharges", "TotalRevenue",
    ]  # SatisfactionScore removed — leakage (see DROP_COLS)

    TARGET_COL = "ChurnLabel"

    def __init__(self):
        self.label_encoders = {}
        self.binary_encoders= {}
        self.scaler         = StandardScaler()
        self.imputer_num    = SimpleImputer(strategy="median")
        self.imputer_cat    = SimpleImputer(strategy="most_frequent")
        self.cat_dims_      = []
        self.fitted_        = False

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Revenue per month
        df["revenue_per_month"] = df["TotalRevenue"] / (df["TenureinMonths"] + 1)

        # Charge ratio
        df["charge_ratio"] = df["MonthlyCharge"] / (df["TotalRevenue"] + 1)

        # Service count
        service_cols = [
            "PhoneService","MultipleLines","InternetService",
            "OnlineSecurity","OnlineBackup","DeviceProtectionPlan",
            "PremiumTechSupport","StreamingTV","StreamingMovies",
            "StreamingMusic","UnlimitedData"
        ]
        df["service_count"] = df[[c for c in service_cols if c in df.columns]].apply(
            lambda row: sum(1 for v in row if str(v) in ("Yes","1",1)), axis=1
        )

        self.NUMERICAL_COLS = list(set(self.NUMERICAL_COLS + [
            "revenue_per_month", "charge_ratio", "service_count"
        ]))
        return df

    def load_and_clean(self, path: str = None) -> pd.DataFrame:
        path = path or TELCO_CSV
        df   = pd.read_csv(path)
        df   = df.drop(columns=[c for c in self.DROP_COLS if c in df.columns])
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df = self._engineer_features(df)
        return df

    def fit_transform(self, df: pd.DataFrame):
        y = (df[self.TARGET_COL].str.strip() == "Yes").astype(int).values

        num_cols = [c for c in self.NUMERICAL_COLS if c in df.columns]
        cat_cols = [c for c in self.CATEGORICAL_COLS if c in df.columns]
        bin_cols = [c for c in self.BINARY_COLS if c in df.columns]

        # Encode binary as 0/1
        X_bin_df = df[bin_cols].copy().astype(str).fillna("No")
        for c in bin_cols:
            X_bin_df[c] = (X_bin_df[c].str.strip() == "Yes").astype(int)
            # Treat gender separately
            if c == "Gender":
                X_bin_df[c] = (X_bin_df[c].astype(str).str.strip() == "Male").astype(int)

        X_num = df[num_cols].copy().astype(float)
        X_num = pd.DataFrame(
            self.imputer_num.fit_transform(X_num), columns=num_cols
        )

        # Combine numerical + binary
        X_num_all = np.hstack([
            self.scaler.fit_transform(X_num),
            X_bin_df.values.astype(float)
        ])

        X_cat_df = df[cat_cols].copy().astype(str).fillna("UNK")
        X_cat_df = pd.DataFrame(
            self.imputer_cat.fit_transform(X_cat_df), columns=cat_cols
        )
        self.cat_dims_ = []
        for c in cat_cols:
            le = LabelEncoder()
            X_cat_df[c] = le.fit_transform(X_cat_df[c])
            self.label_encoders[c] = le
            self.cat_dims_.append(len(le.classes_))

        X_cat = X_cat_df.values.astype(int)
        self.fitted_   = True
        self._num_cols = num_cols
        self._cat_cols = cat_cols
        self._bin_cols = bin_cols

        return X_num_all, X_cat, y

    def transform(self, df: pd.DataFrame):
        assert self.fitted_
        y = (df[self.TARGET_COL].str.strip() == "Yes").astype(int).values

        num_cols = self._num_cols
        cat_cols = self._cat_cols
        bin_cols = self._bin_cols

        X_bin_df = df[bin_cols].copy().astype(str).fillna("No")
        for c in bin_cols:
            X_bin_df[c] = (X_bin_df[c].str.strip() == "Yes").astype(int)
            if c == "Gender":
                X_bin_df[c] = (X_bin_df[c].astype(str).str.strip() == "Male").astype(int)

        X_num = df[num_cols].copy().astype(float)
        X_num = pd.DataFrame(
            self.imputer_num.transform(X_num), columns=num_cols
        )
        X_num_all = np.hstack([
            self.scaler.transform(X_num),
            X_bin_df.values.astype(float)
        ])

        X_cat_df = df[cat_cols].copy().astype(str).fillna("UNK")
        X_cat_df = pd.DataFrame(
            self.imputer_cat.transform(X_cat_df), columns=cat_cols
        )
        for c in cat_cols:
            le = self.label_encoders[c]
            lookup = {cls: i for i, cls in enumerate(le.classes_)}
            X_cat_df[c] = X_cat_df[c].map(lambda x: lookup.get(x, 0)).astype(int)

        X_cat = X_cat_df.values.astype(int)
        return X_num_all, X_cat, y


# ─────────────────────────────────────────────────────────────────────────────
# BANK CUSTOMER CHURN
# ─────────────────────────────────────────────────────────────────────────────

class BankDataProcessor:
    """
    Preprocessing pipeline for Bank Customer Churn dataset.
    Target: churn  (1 = churned, 0 = retained)
    10,000 customers, ~20% churn rate.
    """

    DROP_COLS = ["customer_id"]

    NUMERICAL_COLS = [
        "credit_score", "age", "tenure", "balance",
        "products_number", "estimated_salary",
    ]

    BINARY_COLS = ["credit_card", "active_member"]

    CATEGORICAL_COLS = ["country", "gender"]

    TARGET_COL = "churn"

    def __init__(self):
        self.label_encoders = {}
        self.scaler         = StandardScaler()
        self.imputer_num    = SimpleImputer(strategy="median")
        self.imputer_cat    = SimpleImputer(strategy="most_frequent")
        self.cat_dims_      = []
        self.fitted_        = False

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Balance-to-salary ratio (financial stress proxy)
        df["balance_salary_ratio"] = df["balance"] / (df["estimated_salary"] + 1)
        # Zero-balance flag (customers who keep no funds are at-risk)
        df["is_zero_balance"] = (df["balance"] == 0).astype(int)
        # Age-tenure ratio (young with long tenure = loyal)
        df["tenure_age_ratio"] = df["tenure"] / (df["age"] + 1)
        # Products per tenure (engagement rate)
        df["products_per_tenure"] = df["products_number"] / (df["tenure"] + 1)

        self.NUMERICAL_COLS = list(set(self.NUMERICAL_COLS + [
            "balance_salary_ratio", "tenure_age_ratio", "products_per_tenure",
        ]))
        self.BINARY_COLS = list(set(self.BINARY_COLS + ["is_zero_balance"]))
        return df

    def load_and_clean(self, path: str = None) -> pd.DataFrame:
        path = path or BANK_CSV
        df   = pd.read_csv(path)
        df   = df.drop(columns=[c for c in self.DROP_COLS if c in df.columns])
        df   = self._engineer_features(df)
        return df

    def fit_transform(self, df: pd.DataFrame):
        y = df[self.TARGET_COL].values.astype(int)

        num_cols = [c for c in self.NUMERICAL_COLS if c in df.columns]
        cat_cols = [c for c in self.CATEGORICAL_COLS if c in df.columns]
        bin_cols = [c for c in self.BINARY_COLS if c in df.columns]

        X_bin_df = df[bin_cols].copy().fillna(0).astype(float)

        X_num = df[num_cols].copy().astype(float)
        X_num = pd.DataFrame(
            self.imputer_num.fit_transform(X_num), columns=num_cols
        )
        X_num_all = np.hstack([
            self.scaler.fit_transform(X_num),
            X_bin_df.values
        ])

        X_cat_df = df[cat_cols].copy().astype(str).fillna("UNK")
        X_cat_df = pd.DataFrame(
            self.imputer_cat.fit_transform(X_cat_df), columns=cat_cols
        )
        self.cat_dims_ = []
        for c in cat_cols:
            le = LabelEncoder()
            X_cat_df[c] = le.fit_transform(X_cat_df[c])
            self.label_encoders[c] = le
            self.cat_dims_.append(len(le.classes_))

        X_cat = X_cat_df.values.astype(int)
        self.fitted_   = True
        self._num_cols = num_cols
        self._cat_cols = cat_cols
        self._bin_cols = bin_cols

        return X_num_all, X_cat, y

    def transform(self, df: pd.DataFrame):
        assert self.fitted_
        y = df[self.TARGET_COL].values.astype(int)

        num_cols = self._num_cols
        cat_cols = self._cat_cols
        bin_cols = self._bin_cols

        X_bin_df = df[bin_cols].copy().fillna(0).astype(float)

        X_num = df[num_cols].copy().astype(float)
        X_num = pd.DataFrame(
            self.imputer_num.transform(X_num), columns=num_cols
        )
        X_num_all = np.hstack([
            self.scaler.transform(X_num),
            X_bin_df.values
        ])

        X_cat_df = df[cat_cols].copy().astype(str).fillna("UNK")
        X_cat_df = pd.DataFrame(
            self.imputer_cat.transform(X_cat_df), columns=cat_cols
        )
        for c in cat_cols:
            le = self.label_encoders[c]
            lookup = {cls: i for i, cls in enumerate(le.classes_)}
            X_cat_df[c] = X_cat_df[c].map(lambda x: lookup.get(x, 0)).astype(int)

        X_cat = X_cat_df.values.astype(int)
        return X_num_all, X_cat, y


# ─────────────────────────────────────────────────────────────────────────────
# DATATHON 2019 CORPORATE TRAVEL CHURN
# ─────────────────────────────────────────────────────────────────────────────

class DatathonDataProcessor:
    """
    Preprocessing for the Datathon 2019 corporate travel dataset.
    Files: users.csv, hotels.csv, flights.csv  (no explicit churn label).

    Churn label construction (recency-based):
      Reference date = max date across all activity in the dataset.
      Users with no activity in the last CHURN_DAYS_THRESHOLD days = churned (1).
      Users with recent activity = retained (0).

    Features are aggregated per user from hotels + flights booking history.
    """

    CHURN_PERCENTILE = 70   # users in top (100-70)% least-active = churned (~30% churn)
    TOP_N_COMPANIES      = 20   # top-N companies kept; rest → "Other"
    TOP_N_AGENCIES       = 10

    TARGET_COL = "churned"

    NUMERICAL_COLS = [
        "age", "n_hotel_bookings", "n_flights",
        "total_hotel_spend", "total_flight_spend",
        "avg_hotel_days", "avg_flight_price",
        "avg_flight_distance", "trips_per_month",
        # days_since_last_activity excluded — it directly encodes the churn label (leakage)
    ]

    CATEGORICAL_COLS = ["gender", "company_group", "pref_flight_type", "pref_agency"]

    def __init__(self):
        self.label_encoders = {}
        self.scaler         = StandardScaler()
        self.imputer_num    = SimpleImputer(strategy="median")
        self.imputer_cat    = SimpleImputer(strategy="most_frequent")
        self.cat_dims_      = []
        self.fitted_        = False
        self._top_companies = []
        self._top_agencies  = []

    def load_and_clean(self, data_dir: str = None) -> pd.DataFrame:
        data_dir = data_dir or DATATHON_DIR

        users   = pd.read_csv(os.path.join(data_dir, "users.csv"))
        hotels  = pd.read_csv(os.path.join(data_dir, "hotels.csv"))
        flights = pd.read_csv(os.path.join(data_dir, "flights.csv"))

        # Parse dates
        hotels["date"]  = pd.to_datetime(hotels["date"],  format="%m/%d/%Y", errors="coerce")
        flights["date"] = pd.to_datetime(flights["date"], format="%m/%d/%Y", errors="coerce")

        # Reference date
        ref_date = max(
            hotels["date"].max(),
            flights["date"].max()
        )

        # Last activity per user
        hotel_last  = hotels.groupby("userCode")["date"].max().rename("hotel_last")
        flight_last = flights.groupby("userCode")["date"].max().rename("flight_last")
        last_act = pd.concat([hotel_last, flight_last], axis=1)
        last_act["last_activity"] = last_act[["hotel_last", "flight_last"]].max(axis=1)
        last_act["days_since_last_activity"] = (ref_date - last_act["last_activity"]).dt.days.fillna(999)

        # Churn label — percentile-based: top (100-CHURN_PERCENTILE)% least active = churned
        threshold = np.percentile(
            last_act["days_since_last_activity"].replace(999, np.nan).dropna(), self.CHURN_PERCENTILE
        )
        last_act["churned"] = (last_act["days_since_last_activity"] > threshold).astype(int)

        # Hotel aggregates
        h_agg = hotels.groupby("userCode").agg(
            n_hotel_bookings  = ("travelCode", "count"),
            total_hotel_spend = ("total",       "sum"),
            avg_hotel_days    = ("days",         "mean"),
        ).reset_index()

        # Flight aggregates
        flight_type_mode = (flights.groupby("userCode")["flightType"]
                            .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else "unknown")
                            .rename("pref_flight_type"))
        agency_mode      = (flights.groupby("userCode")["agency"]
                            .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else "unknown")
                            .rename("pref_agency"))

        f_agg = flights.groupby("userCode").agg(
            n_flights          = ("travelCode",  "count"),
            total_flight_spend = ("price",        "sum"),
            avg_flight_price   = ("price",        "mean"),
            avg_flight_distance= ("distance",     "mean"),
        ).reset_index()
        f_agg = f_agg.merge(flight_type_mode.reset_index(), on="userCode", how="left")
        f_agg = f_agg.merge(agency_mode.reset_index(),      on="userCode", how="left")

        # Date range for frequency calculation
        min_date   = min(hotels["date"].min(), flights["date"].min())
        obs_months = max((ref_date - min_date).days / 30.44, 1.0)

        # Merge everything
        df = users.merge(last_act[["days_since_last_activity", "churned"]].reset_index(),
                         left_on="code", right_on="userCode", how="left")
        df = df.merge(h_agg, left_on="code", right_on="userCode", how="left")
        df = df.merge(f_agg, left_on="code", right_on="userCode", how="left")

        # Fill zeros for users with no bookings
        df["n_hotel_bookings"]   = df["n_hotel_bookings"].fillna(0)
        df["n_flights"]          = df["n_flights"].fillna(0)
        df["total_hotel_spend"]  = df["total_hotel_spend"].fillna(0)
        df["total_flight_spend"] = df["total_flight_spend"].fillna(0)
        df["avg_hotel_days"]     = df["avg_hotel_days"].fillna(0)
        df["avg_flight_price"]   = df["avg_flight_price"].fillna(0)
        df["avg_flight_distance"]= df["avg_flight_distance"].fillna(0)
        df["pref_flight_type"]   = df["pref_flight_type"].fillna("unknown")
        df["pref_agency"]        = df["pref_agency"].fillna("unknown")

        # Trip frequency (total trips per month)
        df["trips_per_month"] = (df["n_hotel_bookings"] + df["n_flights"]) / obs_months

        # Users with no activity at all: days_since_last_activity = 999 → definitely churned
        df["churned"] = df["churned"].fillna(1).astype(int)
        print(f"  [datathon] churn threshold: {threshold:.0f} days | churn rate: {df['churned'].mean():.3f}")

        # Company grouping: top-N keep, rest → "Other"
        self._top_companies = (df["company"].value_counts()
                               .head(self.TOP_N_COMPANIES).index.tolist())
        df["company_group"] = df["company"].apply(
            lambda x: x if x in self._top_companies else "Other"
        )
        self._top_agencies = (df["pref_agency"].value_counts()
                              .head(self.TOP_N_AGENCIES).index.tolist())
        df["pref_agency"] = df["pref_agency"].apply(
            lambda x: x if x in self._top_agencies else "Other"
        )

        print(f"[datathon] {len(df):,} users | churn rate = {df['churned'].mean():.3f}")
        return df

    def fit_transform(self, df: pd.DataFrame):
        y = df[self.TARGET_COL].values.astype(int)

        num_cols = [c for c in self.NUMERICAL_COLS if c in df.columns]
        cat_cols = [c for c in self.CATEGORICAL_COLS if c in df.columns]

        X_num = df[num_cols].copy().astype(float)
        X_num = pd.DataFrame(
            self.imputer_num.fit_transform(X_num), columns=num_cols
        )
        X_num = self.scaler.fit_transform(X_num)

        X_cat_df = df[cat_cols].copy().astype(str).fillna("UNK")
        X_cat_df = pd.DataFrame(
            self.imputer_cat.fit_transform(X_cat_df), columns=cat_cols
        )
        self.cat_dims_ = []
        for c in cat_cols:
            le = LabelEncoder()
            X_cat_df[c] = le.fit_transform(X_cat_df[c])
            self.label_encoders[c] = le
            self.cat_dims_.append(len(le.classes_))

        X_cat = X_cat_df.values.astype(int)
        self.fitted_   = True
        self._num_cols = num_cols
        self._cat_cols = cat_cols

        return X_num, X_cat, y

    def transform(self, df: pd.DataFrame):
        assert self.fitted_
        y = df[self.TARGET_COL].values.astype(int)

        num_cols = self._num_cols
        cat_cols = self._cat_cols

        X_num = df[num_cols].copy().astype(float)
        X_num = pd.DataFrame(
            self.imputer_num.transform(X_num), columns=num_cols
        )
        X_num = self.scaler.transform(X_num)

        X_cat_df = df[cat_cols].copy().astype(str).fillna("UNK")
        X_cat_df = pd.DataFrame(
            self.imputer_cat.transform(X_cat_df), columns=cat_cols
        )
        for c in cat_cols:
            le = self.label_encoders[c]
            lookup = {cls: i for i, cls in enumerate(le.classes_)}
            X_cat_df[c] = X_cat_df[c].map(lambda x: lookup.get(x, 0)).astype(int)

        X_cat = X_cat_df.values.astype(int)
        return X_num, X_cat, y


# ─────────────────────────────────────────────────────────────────────────────
# HOTEL FEATURE GROUP MAPPING  (for Semantic Group Attention)
# ─────────────────────────────────────────────────────────────────────────────

def get_hotel_feature_groups(num_cols: list, cat_cols: list) -> dict:
    """
    Returns semantic feature groups for the hotel dataset.

    Output: { group_name : [feature_indices] }
    Indices are 0-based into the concatenated [num_cols + cat_cols] array,
    which matches the token order produced by FeatureTokenizer.

    Groups are defined from hotel CRM domain knowledge:
      Temporal      — booking horizon & arrival timing features
      Stay_Pattern  — length-of-stay & guest composition
      Guest_History — loyalty & past booking behaviour
      Commercial    — pricing, distribution channel & deposit policy
      Property_Room — hotel type, room assignment & service amenities

    Used by HIFTTransformer's Semantic Group Attention module.
    """
    all_cols = list(num_cols) + list(cat_cols)
    col_idx  = {c: i for i, c in enumerate(all_cols)}

    GROUP_DEFS = {
        "Temporal": [
            "lead_time", "arrival_date_year",
            "arrival_date_week_number", "arrival_date_day_of_month",
            "days_in_waiting_list",
            "month_sin", "month_cos", "week_sin", "week_cos",
            "lead_per_night", "arrival_date_month",
            "long_lead_no_commit",          # NEW: high-risk horizon flag
        ],
        "Stay_Pattern": [
            "stays_in_weekend_nights", "stays_in_week_nights",
            "total_nights", "has_weekend_nights",
            "adults", "children", "babies",
            "party_size",                   # NEW: total occupants
            "is_family",                    # NEW: family booking flag
            "weekend_proportion",           # NEW: leisure vs business ratio
        ],
        "Guest_History": [
            "is_repeated_guest", "previous_cancellations",
            "previous_bookings_not_canceled", "booking_changes",
            "cancellation_risk_ratio",      # NEW: chronic canceller ratio
            "loyalty_score",               # NEW: repeat × no-cancel score
            "booking_instability",         # NEW: changes per lead time
        ],
        "Commercial": [
            "adr", "revenue_proxy", "revenue_per_night",  # NEW: per-night value
            "adr_per_person",              # NEW: price per occupant
            "deposit_type", "market_segment",
            "distribution_channel", "customer_type",
        ],
        "Property_Room": [
            "hotel", "meal",
            "reserved_room_type", "assigned_room_type",
            "room_mismatch", "required_car_parking_spaces",
            "total_of_special_requests",
            "engagement_ratio",            # NEW: requests per person
        ],
    }

    groups = {}
    for g_name, g_cols in GROUP_DEFS.items():
        indices = [col_idx[c] for c in g_cols if c in col_idx]
        if indices:
            groups[g_name] = indices
    return groups


def _build_groups(all_cols: list, group_defs: dict) -> dict:
    """Map named feature groups to 0-based token indices over all_cols."""
    col_idx = {c: i for i, c in enumerate(all_cols)}
    groups = {}
    for g_name, g_cols in group_defs.items():
        idx = [col_idx[c] for c in g_cols if c in col_idx]
        if idx:
            groups[g_name] = idx
    return groups


def get_telco_feature_groups(num_cols: list, bin_cols: list, cat_cols: list) -> dict:
    """
    Semantic feature groups for the IBM Telco dataset.

    Token order produced by TelcoDataProcessor is  num_cols + bin_cols + cat_cols
    (numerical are scaled, binary appended into the numerical tensor, then
    categorical). Indices are 0-based into that concatenation, matching the
    FeatureTokenizer token order consumed by Semantic Group Attention.

    The five groups mirror the hotel design so BPAC + SGA are genuinely active
    off-hotel (customer-lifecycle analogue of the booking horizon):
      Tenure_Contract — lifecycle stage & contractual commitment
      Services        — subscribed products & usage
      Charges         — billing, revenue & refunds
      Demographics    — customer attributes
      Engagement      — referral / advocacy behaviour
    """
    all_cols = list(num_cols) + list(bin_cols) + list(cat_cols)
    GROUP_DEFS = {
        "Tenure_Contract": [
            "TenureinMonths", "Contract", "PaymentMethod",
            "PaperlessBilling", "Offer",
        ],
        "Services": [
            "PhoneService", "MultipleLines", "InternetService", "InternetType",
            "OnlineSecurity", "OnlineBackup", "DeviceProtectionPlan",
            "PremiumTechSupport", "StreamingTV", "StreamingMovies",
            "StreamingMusic", "UnlimitedData", "AvgMonthlyGBDownload",
            "service_count",
        ],
        "Charges": [
            "MonthlyCharge", "TotalCharges", "TotalRevenue",
            "AvgMonthlyLongDistanceCharges", "TotalLongDistanceCharges",
            "TotalRefunds", "TotalExtraDataCharges",
            "revenue_per_month", "charge_ratio",
        ],
        "Demographics": [
            "Age", "Gender", "SeniorCitizen", "Married",
            "Dependents", "NumberofDependents",
        ],
        "Engagement": [
            "Number_of_Referrals", "ReferredaFriend",
        ],
    }
    return _build_groups(all_cols, GROUP_DEFS)


def get_bank_feature_groups(num_cols: list, bin_cols: list, cat_cols: list) -> dict:
    """
    Semantic feature groups for the Bank Customer Churn dataset.

    Token order: num_cols + bin_cols + cat_cols (see BankDataProcessor).
      Tenure_Relationship — lifecycle stage & product engagement
      Financial           — balance, salary, credit standing
      Demographics        — customer attributes
    """
    all_cols = list(num_cols) + list(bin_cols) + list(cat_cols)
    GROUP_DEFS = {
        "Tenure_Relationship": [
            "tenure", "products_number", "products_per_tenure",
            "tenure_age_ratio", "active_member",
        ],
        "Financial": [
            "balance", "estimated_salary", "balance_salary_ratio",
            "is_zero_balance", "credit_score", "credit_card",
        ],
        "Demographics": [
            "age", "gender", "country",
        ],
    }
    return _build_groups(all_cols, GROUP_DEFS)


# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED DATA PREPARATION API
# ─────────────────────────────────────────────────────────────────────────────

_MONTHS = {m: i for i, m in enumerate(
    ["January","February","March","April","May","June",
     "July","August","September","October","November","December"], start=1)}


def _add_hotel_arrival_order(df: pd.DataFrame) -> pd.DataFrame:
    """Monotone arrival-date key for temporal ordering (year*10000+month*100+day)."""
    df = df.copy()
    yr  = pd.to_numeric(df.get("arrival_date_year"), errors="coerce")
    mo  = df.get("arrival_date_month")
    if mo is not None and mo.dtype == object:
        mo = mo.map(_MONTHS)
    mo  = pd.to_numeric(mo, errors="coerce")
    day = pd.to_numeric(df.get("arrival_date_day_of_month"), errors="coerce")
    df["_arrival_order"] = (yr.fillna(0) * 10000 + mo.fillna(0) * 100 + day.fillna(0))
    return df


def prepare_dataset(
    dataset: str = "hotel",
    use_smote: bool = True,
    batch_size: int = 512,
    return_cv: bool = False,
    seed: int = None,
    split_mode: str = "random",
):
    """
    Parameters
    ----------
    dataset   : "hotel" | "telco"
    use_smote : apply SMOTE on training set to address class imbalance
    batch_size: DataLoader batch size
    return_cv : if True also return StratifiedKFold splits on full data

    Returns
    -------
    loaders      : dict with "train" / "val" / "test" DataLoaders
    meta         : dict with cat_dims, num_features, pos_weight, processor
    cv_splits    : list of (train_idx, val_idx) if return_cv=True
    """
    if dataset == "hotel":
        proc = HotelDataProcessor()
    elif dataset == "telco":
        proc = TelcoDataProcessor()
    elif dataset == "bank":
        proc = BankDataProcessor()
    elif dataset == "datathon":
        proc = DatathonDataProcessor()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    seed = SEED if seed is None else seed

    df = proc.load_and_clean()
    print(f"[{dataset}] Loaded {len(df):,} rows  |  "
          f"churn rate = {df[proc.TARGET_COL].map({'Yes':1,'No':0}).fillna(df[proc.TARGET_COL]).mean():.3f}"
          f"  |  split={split_mode} seed={seed}")

    if split_mode == "hotelwise" and dataset == "hotel" and "hotel" in df.columns:
        # Grouped holdout by property: train/val on City Hotel, test on the
        # unseen Resort Hotel (R2 Major7 — generalisation to a new property).
        df_test = df[df["hotel"] == "Resort Hotel"].reset_index(drop=True)
        df_dev  = df[df["hotel"] != "Resort Hotel"].reset_index(drop=True)
        if len(df_test) == 0 or len(df_dev) == 0:   # fallback if labels differ
            df_test = df[df["hotel"] == df["hotel"].unique()[0]].reset_index(drop=True)
            df_dev  = df[df["hotel"] != df["hotel"].unique()[0]].reset_index(drop=True)
        df_train, df_val = train_test_split(
            df_dev, test_size=VAL_SIZE / (1 - TEST_SIZE),
            stratify=df_dev[proc.TARGET_COL], random_state=seed,
        )
    elif split_mode == "temporal" and dataset == "hotel":
        # Time-ordered split: earliest arrivals train, latest arrivals test.
        # Tests robustness to distribution shift (R2 Major7) beyond random split.
        df = _add_hotel_arrival_order(df)
        df = df.sort_values("_arrival_order", kind="mergesort").reset_index(drop=True)
        n = len(df)
        n_test = int(round(TEST_SIZE * n))
        n_val  = int(round(VAL_SIZE  * n))
        df_test  = df.iloc[n - n_test:]
        df_val   = df.iloc[n - n_test - n_val: n - n_test]
        df_train = df.iloc[: n - n_test - n_val]
        for _d in (df_train, df_val, df_test):
            _d.drop(columns=["_arrival_order"], inplace=True, errors="ignore")
    else:
        # First split: hold-out test
        df_train_val, df_test = train_test_split(
            df, test_size=TEST_SIZE, stratify=df[proc.TARGET_COL],
            random_state=seed
        )
        # Second split: train / val
        df_train, df_val = train_test_split(
            df_train_val,
            test_size=VAL_SIZE / (1 - TEST_SIZE),
            stratify=df_train_val[proc.TARGET_COL],
            random_state=seed,
        )

    # Fit on train, transform all
    X_num_tr, X_cat_tr, y_tr = proc.fit_transform(df_train)
    X_num_va, X_cat_va, y_va = proc.transform(df_val)
    X_num_te, X_cat_te, y_te = proc.transform(df_test)

    # Class imbalance info
    pos_count = y_tr.sum()
    neg_count = len(y_tr) - pos_count
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32)

    print(f"  Train {len(y_tr):,} | Val {len(y_va):,} | Test {len(y_te):,}")
    print(f"  Churn in train: {pos_count} / {len(y_tr)} ({100*pos_count/len(y_tr):.1f}%)")

    # SMOTE on training set (numerical only for tree models; full for DL)
    if use_smote and pos_count < neg_count * 0.5:
        X_flat = np.hstack([X_num_tr, X_cat_tr.astype(float)])
        sm = SMOTE(random_state=seed, k_neighbors=5)
        X_res, y_res = sm.fit_resample(X_flat, y_tr)
        n_num = X_num_tr.shape[1]
        X_num_tr = X_res[:, :n_num]
        X_cat_tr = X_res[:, n_num:].astype(int)
        y_tr     = y_res.astype(int)
        print(f"  After SMOTE: {y_tr.sum()} / {len(y_tr)}")

    # PyTorch DataLoaders
    ds_train = ChurnDataset(X_num_tr, X_cat_tr, y_tr)
    ds_val   = ChurnDataset(X_num_va, X_cat_va, y_va)
    ds_test  = ChurnDataset(X_num_te, X_cat_te, y_te)

    loaders = {
        "train": DataLoader(ds_train, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=False),
        "val"  : DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False),
        "test" : DataLoader(ds_test,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False),
    }

    meta = {
        "cat_dims"    : proc.cat_dims_,
        "num_features": X_num_tr.shape[1],
        "pos_weight"  : pos_weight,
        "processor"   : proc,
        # Raw arrays for sklearn models
        "X_num_tr": X_num_tr, "X_cat_tr": X_cat_tr, "y_tr": y_tr,
        "X_num_va": X_num_va, "X_cat_va": X_cat_va, "y_va": y_va,
        "X_num_te": X_num_te, "X_cat_te": X_cat_te, "y_te": y_te,
    }

    if return_cv:
        # For statistical tests: CV folds over the concatenated train+val arrays
        # (works for every split_mode; no dependency on the raw train_val frame).
        X_all = np.hstack([
            np.vstack([X_num_tr, X_num_va]),
            np.vstack([X_cat_tr, X_cat_va]).astype(float)
        ])
        y_all = np.concatenate([y_tr, y_va])
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
        cv_splits = list(skf.split(X_all, y_all))
        meta["X_all_tv"]  = X_all
        meta["y_all_tv"]  = y_all
        meta["X_num_all"] = np.vstack([X_num_tr, X_num_va])
        meta["X_cat_all"] = np.vstack([X_cat_tr, X_cat_va])
        return loaders, meta, cv_splits

    return loaders, meta
