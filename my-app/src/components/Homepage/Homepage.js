import React from 'react';
import { Link } from 'react-router-dom';
import './Homepage.css';

function HomePage() {
    return (
        <div className="homepage">
            <main className="main-content">
                <section className="intro">
                    <h2>Welcome to OptiPortfolio</h2>
                    <p>Our Deep Learning models are designed to help you optimize a long-only portfolio.</p>
                </section>
                
                <div className="bot-cards">
                    <Link to="/configure-bot" className="bot-card-link">
                        <div className="bot-card">
                            <h3>OptiLSTM</h3>
                            <p>OptiLSTM uses a parameter-tuned LSTM to help obtain the best possible Sharpe Ratio, providing a portfolio with ideal risk-adjusted returns.</p>
                        </div>
                    </Link>

                    <Link to="/configure-bot" className="bot-card-link">
                        <div className="bot-card">
                            <h3>Bot Beta</h3>
                            <p>Beta Bot focuses on minimizing risk while ensuring steady growth.</p>
                        </div>
                    </Link>

                    <Link to="/configure-bot" className="bot-card-link">
                        <div className="bot-card">
                            <h3>Bot Gamma</h3>
                            <p>Gamma Bot uses AI to analyze market trends for optimal investment decisions.</p>
                        </div>
                    </Link>
                </div>
            </main>
        </div>
    );
}

export default HomePage;
