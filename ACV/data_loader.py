import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ACVDataLoader:
    """Loads and preprocesses ACV train/test data files."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.train_dir = self.data_dir / "Train"
        self.test_dir = self.data_dir / "Test"
        self.labels_path = self.data_dir / "Train_Labels.csv"
        
    def load_labels(self) -> pd.DataFrame:
        """Load training labels."""
        return pd.read_csv(self.labels_path)
    
    def load_case(self, filepath) -> pd.DataFrame:
        """Load a single case file (prefers parquet for speed)."""
        filepath = Path(filepath)
        # Try parquet first
        parquet_path = filepath.with_suffix('.parquet')
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
        else:
            df = pd.read_excel(filepath)
        df['Time'] = pd.to_datetime(df['Time'])
        df = df.sort_values('Time').reset_index(drop=True)
        return df
    
    def parse_car_columns(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Parse column names to identify cars and their parameters."""
        car_params = {}
        for col in df.columns:
            if col.startswith('Car ') and ' - ' in col:
                car_id, param = col.split(' - ', 1)
                car_id = car_id.replace('Car ', '')
                if car_id not in car_params:
                    car_params[car_id] = []
                car_params[car_id].append(param)
        return car_params
    
    def get_car_dataframes(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Split dataframe into per-car dataframes."""
        car_params = self.parse_car_columns(df)
        id_cols = ['Car model', 'Train number', 'Time']
        
        car_dfs = {}
        for car_id, params in car_params.items():
            car_cols = id_cols + [f'Car {car_id} - {p}' for p in params]
            car_df = df[car_cols].copy()
            car_df.columns = id_cols + params
            car_dfs[car_id] = car_df
            
        return car_dfs
    
    def load_all_train_cases(self) -> List[Tuple[str, pd.DataFrame, str]]:
        """Load all training cases with labels.
        
        Returns:
            List of (case_id, dataframe, faulty_car_id)
        """
        labels_df = self.load_labels()
        cases = []
        
        for _, row in labels_df.iterrows():
            filepath = self.train_dir / row['filename']
            df = self.load_case(filepath)
            cases.append((row['filename'], df, row['faulty_car']))
            
        return cases
    
    def load_test_case(self) -> pd.DataFrame:
        """Load test case."""
        test_files = list(self.test_dir.glob("*.parquet"))
        if not test_files:
            test_files = list(self.test_dir.glob("*.xlsx"))
        if not test_files:
            raise FileNotFoundError("No test file found")
        return self.load_case(test_files[0])
    
    def get_case_info(self, df: pd.DataFrame) -> Dict:
        """Get metadata about a case."""
        car_params = self.parse_car_columns(df)
        return {
            'n_rows': len(df),
            'n_cars': len(car_params),
            'cars': sorted(car_params.keys()),
            'params_per_car': {k: len(v) for k, v in car_params.items()},
            'all_params': {k: sorted(v) for k, v in car_params.items()},
            'time_range': (df['Time'].min(), df['Time'].max()),
            'sampling_interval': (df['Time'].diff().median().total_seconds() 
                                  if len(df) > 1 else None)
        }


def load_data(data_dir: str = "data") -> Dict:
    """Convenience function to load all data."""
    loader = ACVDataLoader(data_dir)
    train_cases = loader.load_all_train_cases()
    test_df = loader.load_test_case()
    test_info = loader.get_case_info(test_df)
    
    return {
        'train_cases': train_cases,
        'test_df': test_df,
        'test_info': test_info,
        'loader': loader
    }


if __name__ == "__main__":
    data = load_data()
    print("Train cases:")
    for case_id, df, faulty in data['train_cases']:
        info = data['loader'].get_case_info(df)
        print(f"  {case_id}: faulty={faulty}, rows={info['n_rows']}, cars={info['cars']}, params_per_car={info['params_per_car']}")
    print(f"\nTest case: rows={data['test_info']['n_rows']}, cars={data['test_info']['cars']}")