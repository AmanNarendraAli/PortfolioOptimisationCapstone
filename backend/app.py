from flask import Flask, request, jsonify
from datetime import datetime, timedelta
import yfinance as yf
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS to allow requests from the frontend

@app.route('/api/configure-bot', methods=['POST'])
def configure_bot():
    data = request.json
    budget = data.get('budget')
    volatility = data.get('volatility')
    tickers = data.get('tickers')

    # Validate budget and tickers
    if not budget or not tickers or not isinstance(tickers, list) or len(tickers) == 0:
        return jsonify({"error": "Invalid budget or tickers"}), 400

    # Set date range dynamically: current date and 2 years prior
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=2*365)).strftime('%Y-%m-%d')

    # Download data using yfinance for the given tickers and date range
    try:
        # Download historical data for the selected tickers
        data = yf.download(tickers, start=start_date, end=end_date)
        # At this point, 'data' is in DataFrame format, ideal for further processing
        
        # Placeholder for further processing
        # Here you could call your ML model or perform calculations with 'data'
        # Example: result = some_model.predict(data, budget)
        
        # Convert only necessary data for frontend visualization, if required
        data_sample = data.reset_index().head(5).to_dict(orient="records")  # Example of sampling data for frontend preview
    except Exception as e:
        print(f"Error downloading data: {e}")
        return jsonify({"error": "Failed to fetch data for the given tickers."}), 500

    # Debug output
    print(f"Budget: {budget}")
    print(f"Target Volatility: {volatility}")
    print(f"Tickers: {tickers}")
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    print("Downloaded Data Sample:", data.head())  # Print first few rows for verification

    # Return a minimal preview of data for frontend display purposes
    return jsonify({
        "budget": budget,
        "volatility": volatility,
        "tickers": tickers,
        "start_date": start_date,
        "end_date": end_date,
        # "data_sample": data_sample  # Provide only a sample if necessary
    })

if __name__ == '__main__':
    app.run(debug=True)
