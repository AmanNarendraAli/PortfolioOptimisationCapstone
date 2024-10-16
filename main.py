from Model import Model
import logging
#code currently runs on QuantConnect platform. Should aim to move it towards a local environment, could maybe have 2 versions
class OptimizedUncoupledAutosequencers(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2015, 10, 1)  # Set Start Date
        self.SetEndDate(2020, 10, 1)
        self.SetCash(100000)  # Set Strategy Cash
        
        tickers = ['VTI', 'AGG', 'DBC', 'VIXY']
        
        for ticker in tickers:
            self.AddEquity(ticker, Resolution.Daily)
        
        n_periods = 51
        
        self.data = RollingWindow[Slice](n_periods) 
        
        self.Train(self.DateRules.MonthStart('VTI'), self.TimeRules.Midnight, self.Rebalance)
        
        self.model = None
        
        self.SetWarmup(n_periods)
        
        self.prev_day = -1
        
    def OnData(self, data):
        
        if self.prev_day != self.Time.day:
            self.data.Add(data)
        
        self.prev_day = self.Time.day

    def Rebalance(self):
        
        if not self.data.IsReady:
            return
        
        try:
            
            data = self.PandasConverter.GetDataFrame(self.data).iloc[::-1]
        except:
            return
     
        
        data = data['close'].unstack(level=0)
        
    
        if len(data) < self.data.Count:
            return
        
        tickers = [symbol.split(' ')[0] for symbol in data.columns]
        
        if self.model is None:
            self.model = Model()
        
        allocations = self.model.get_allocations(data)
        self.Log(f'Portfolio Allocations: {allocations}')
        
        for ticker, allocation in zip(tickers, allocations):
            self.SetHoldings(ticker, allocation)