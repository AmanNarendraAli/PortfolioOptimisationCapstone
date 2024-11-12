import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import './TickerPage.css';

function Tickerpage() {
    const [tickers, setTickers] = useState(['']);
    const [budget, setBudget] = useState('');
    const [volatility, setVolatility] = useState('');

    // Handle adding a new ticker input
    const addTicker = () => {
        setTickers([...tickers, '']);
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

    // Handle form submission
    const handleSubmit = async () => {
        const payload = {
            budget,
            tickers: tickers.filter(ticker => ticker) // Remove empty tickers
        };

        try {
            const response = await fetch('http://127.0.0.1:5000/api/configure-bot', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error('Failed to submit data');
            }

            const data = await response.json();
            console.log('Response from backend:', data);
        } catch (error) {
            console.error('Error:', error);
        }
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

            <label htmlFor="volatility">Target Volatility</label>
            <input
                type="number"
                id="volatility"
                placeholder="Enter your budget"
                value={volatility}
                onChange={(e) => setVolatility(e.target.value)}
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
            <Link to="/processing"> 
                <button type="button" className="submit-button" onClick={handleSubmit}>
                    Submit
                </button>
            </Link>   
        </div>
    );
}

export default Tickerpage;
