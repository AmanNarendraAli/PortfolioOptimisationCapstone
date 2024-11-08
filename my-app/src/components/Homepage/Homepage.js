import React from 'react';
import './Homepage.css';
import RobotImage from "./robot.png";
function HomePage() {
    return (
        <div className="homepage">
            {/* Main Content Body */}
            <main className="main-content">
                <section className="intro">
                    <h2>Welcome to OptiPortfolio</h2>
                    <p>Our Deep Learning models are designed to help you optimise a long-only portfolio.</p>
                </section>
                
                <div className="bot-cards">
                    <div className="bot-card">
                        <h3>OptiLSTM</h3>
                        <p>OptiLSTM uses a parameter-tuned LSTM to help obtain the best possible Sharpe Ratio, providing a portfolio with ideal risk-adjusted returns.</p>
                    </div>
                    <div className="bot-card">
                        <h3>Bot Beta</h3>
                        <p>Beta Bot focuses on minimizing risk while ensuring steady growth.</p>
                    </div>
                    <div className="bot-card">
                        <h3>Bot Gamma</h3>
                        <p>Gamma Bot uses AI to analyze market trends for optimal investment decisions.</p>
                    </div>
                </div>
            </main>
        </div>
    );
}

export default HomePage;
