import React, { useState } from 'react';
import './TickerPage.css';

function Tickerpage() {
    const [tickers, setTickers] = useState(['']); // Start with one empty ticker input
    const [budget, setBudget] = useState('');

    // Handle adding a new ticker input
    const addTicker = () => {
        setTickers([...tickers, '']); // Add a new empty ticker input
    };

    // Handle removing a ticker, ensuring there's at least one
    const removeTicker = (index) => {
        if (tickers.length > 1) {
            setTickers(tickers.filter((_, i) => i !== index));
        }
    };

    // Handle changing ticker input
    const handleTickerChange = (index, value) => {
        const updatedTickers = [...tickers];
        updatedTickers[index] = value;
        setTickers(updatedTickers);
    };

    return (
        <div className="configure-bot">
            <h2>Configure Your Bot</h2>
            
            <label htmlFor="budget">Overall Budget</label>
            <input
                type="number"
                id="budget"
                placeholder="Enter your budget"
                value={budget}
                onChange={(e) => setBudget(e.target.value)}
            />

            <label>Tickers</label>
            <div className="ticker-list">
                {tickers.map((ticker, index) => (
                    <div key={index} className="ticker-item">
                        <input
                            type="text"
                            placeholder="Enter ticker"
                            value={ticker}
                            onChange={(e) => handleTickerChange(index, e.target.value)}
                        />
                        {tickers.length > 1 && (
                            <button
                                type="button"
                                className="remove-ticker"
                                onClick={() => removeTicker(index)}
                            >
                                ×
                            </button>
                        )}
                    </div>
                ))}
            </div>

            <button type="button" className="add-ticker" onClick={addTicker}>
                Add Ticker
            </button>

            <button type="button" className="submit-button">
                Submit
            </button>
        </div>
    );
}

export default Tickerpage;
