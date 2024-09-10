import yaml
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Union
from datetime import date, datetime, timedelta

class UniverseUtils:
    @staticmethod
    def generate_universe_from_index_rics(index_ric: str, date: date) -> List[str]:
        index_constituents = {
            "SPX": ["AAPL", "MSFT", "AMZN", "GOOGL", "FB"],
            "UKX": ["HSBA", "BP", "GSK", "AZN", "ULVR"],
        }
        return index_constituents.get(index_ric, [])

    @staticmethod
    def generate_rolling_beta(returns: pd.DataFrame, market_returns: pd.Series, window: int = 252) -> pd.DataFrame:
        rolling_cov = returns.rolling(window=window).cov(market_returns)
        rolling_var = market_returns.rolling(window=window).var()
        return rolling_cov.div(rolling_var, axis=0)

    @staticmethod
    def generate_adv_df(volume: pd.DataFrame, price: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        return (volume * price).rolling(window=window).mean()

    @staticmethod
    def generate_returns(prices: pd.DataFrame, method: str = 'log') -> pd.DataFrame:
        if method == 'log':
            return np.log(prices / prices.shift(1))
        elif method == 'simple':
            return prices.pct_change()
        else:
            raise ValueError("Method must be 'log' or 'simple'")

    @staticmethod
    def generate_universe_over_time(start_date: date, end_date: date, frequency: str = 'M', 
                                    base_universe: List[str] = None) -> Dict[date, List[str]]:
        date_range = pd.date_range(start_date, end_date, freq=frequency)
        universe_dict = {}
        
        if base_universe is None:
            base_universe = ["AAPL", "MSFT", "AMZN", "GOOGL", "FB"]
        
        for d in date_range:
            universe = base_universe.copy()
            remove_count = np.random.randint(0, 3)
            if remove_count > 0 and len(universe) > remove_count:
                universe = list(np.random.choice(universe, len(universe) - remove_count, replace=False))
            universe_dict[d.date()] = universe
        
        return universe_dict

@dataclass
class Category:
    name: str
    value: str

@dataclass
class Universe:
    id: str
    name: str
    start_date: date
    end_date: date
    is_list: bool
    universe_list: List[str] = field(default_factory=list)
    universe_df: Optional[pd.DataFrame] = None
    hedge_instruments: List[str] = field(default_factory=list)
    properties: dict = field(default_factory=dict)
    category: Optional[Category] = None
    countries: List[str] = field(default_factory=list)

    def generate_rolling_beta(self, market_returns: pd.Series, window: int = 252) -> pd.DataFrame:
        if self.universe_df is None:
            raise ValueError("Universe DataFrame is not set")
        return UniverseUtils.generate_rolling_beta(self.universe_df, market_returns, window)

    def generate_adv(self, volume: pd.DataFrame, price: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        return UniverseUtils.generate_adv_df(volume, price, window)

    def generate_returns(self, prices: pd.DataFrame, method: str = 'log') -> pd.DataFrame:
        return UniverseUtils.generate_returns(prices, method)

    def update_universe_over_time(self, frequency: str = 'M'):
        self.universe_df = pd.DataFrame.from_dict(UniverseUtils.generate_universe_over_time(
            self.start_date, self.end_date, frequency, self.universe_list
        ), orient='index')

    @classmethod
    def from_index_ric(cls, id: str, name: str, index_ric: str, start_date: date, end_date: date):
        universe_list = UniverseUtils.generate_universe_from_index_rics(index_ric, start_date)
        return cls(id=id, name=name, start_date=start_date, end_date=end_date, 
                   is_list=True, universe_list=universe_list)

class UniverseGenerator:
    def __init__(self, config_path: str):
        self.universes: List[Universe] = []
        self.config = self.load_config(config_path)

    def load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def generate_universes(self):
        for universe_config in self.config['universes']:
            self.generate_universe_from_config(universe_config)

    def generate_universe_from_config(self, config: dict):
        id = config['id']
        name = config['name']
        start_date = datetime.strptime(config['start_date'], '%Y-%m-%d').date()
        end_date = datetime.strptime(config['end_date'], '%Y-%m-%d').date()
        is_list = config['type'] == 'list'

        if 'index_ric' in config:
            universe = Universe.from_index_ric(id, name, config['index_ric'], start_date, end_date)
        else:
            universe_data = self.generate_universe_data(config['source'], start_date, end_date)
            category = None
            if 'category' in config:
                category = Category(
                    name=config['category']['name'],
                    value=config['category']['value']
                )
            universe = Universe(
                id=id,
                name=name,
                start_date=start_date,
                end_date=end_date,
                is_list=is_list,
                universe_list=universe_data if is_list else universe_data.columns.tolist(),
                universe_df=None if is_list else universe_data,
                category=category,
                countries=config.get('countries', [])
            )

        if config.get('generate_time_series', False):
            universe.update_universe_over_time(config.get('frequency', 'M'))

        self.universes.append(universe)

    def generate_universe_data(self, source_config: dict, start_date: date, end_date: date) -> Union[List[str], pd.DataFrame]:
        source_type = source_config['type']

        if source_type == 'static':
            return source_config['data']
        elif source_type == 'random':
            return self.generate_random_df(source_config['symbols'], start_date, end_date)
        elif source_type == 'file':
            return self.load_from_file(source_config['path'])
        else:
            raise ValueError(f"Unknown source type: {source_type}")

    def generate_random_df(self, symbols: List[str], start_date: date, end_date: date) -> pd.DataFrame:
        date_range = pd.date_range(start_date, end_date, freq='D')
        df = pd.DataFrame(index=date_range, columns=symbols)
        for symbol in symbols:
            df[symbol] = np.random.randint(0, 2, size=len(date_range))
        return df

    def load_from_file(self, file_path: str) -> List[str]:
        return pd.read_csv(file_path, header=None, squeeze=True).tolist()

    def generate(self):
        return self.universes

# Usage example
if __name__ == "__main__":
    # First, create a YAML config file
    yaml_content = """
    universes:
      - id: TECH_2023_DF
        name: Technology Sector 2023 (DataFrame)
        start_date: 2023-01-01
        end_date: 2023-12-31
        type: dataframe
        source:
          type: random
          symbols: ["AAPL", "GOOGL", "MSFT", "AMZN", "FB"]
        category:
          name: sector
          value: Technology
        countries: ["USA"]
        generate_time_series: true
        frequency: W

      - id: SP500_2023
        name: S&P 500 2023
        start_date: 2023-01-01
        end_date: 2023-12-31
        type: list
        index_ric: SPX
        countries: ["USA"]
    """

    with open('universe_config.yaml', 'w') as file:
        file.write(yaml_content)

    # Now use the config to generate universes
    generator = UniverseGenerator('universe_config.yaml')
    generator.generate_universes()

    # Get a specific universe
    tech_universe = next(u for u in generator.universes if u.id == 'TECH_2023_DF')

    # Generate market returns (this would typically come from your data source)
    market_returns = pd.Series(np.random.randn(252), index=pd.date_range(tech_universe.start_date, periods=252, freq='B'))

    # Calculate rolling beta
    rolling_beta = tech_universe.generate_rolling_beta(market_returns)

    # Generate ADV (assuming you have volume and price data)
    volume = pd.DataFrame(np.random.randint(100000, 1000000, size=(252, len(tech_universe.universe_list))), 
                          columns=tech_universe.universe_list, 
                          index=pd.date_range(tech_universe.start_date, periods=252, freq='B'))
    price = pd.DataFrame(np.random.randint(50, 200, size=(252, len(tech_universe.universe_list))), 
                         columns=tech_universe.universe_list, 
                         index=pd.date_range(tech_universe.start_date, periods=252, freq='B'))
    adv = tech_universe.generate_adv(volume, price)

    # Generate returns
    returns = tech_universe.generate_returns(price)

    print(f"Rolling Beta for {tech_universe.name}:")
    print(rolling_beta.head())
    print(f"\nADV for {tech_universe.name}:")
    print(adv.head())
    print(f"\nReturns for {tech_universe.name}:")
    print(returns.head())
    print(f"\nUniverse over time for {tech_universe.name} (first 3 entries):")
    print(tech_universe.universe_df.head(3))

    # Print info for SP500 universe
    sp500_universe = next(u for u in generator.universes if u.id == 'SP500_2023')
    print(f"\nSP500 Universe:")
    print(f"Name: {sp500_universe.name}")
    print(f"Start Date: {sp500_universe.start_date}")
    print(f"End Date: {sp500_universe.end_date}")
    print(f"Constituents: {sp500_universe.universe_list}")