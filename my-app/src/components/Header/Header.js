import React from 'react';
import { Link } from 'react-router-dom';
import './Header.css';
import RobotImage from './robot.png';
function Header() {
    return (
        <header className="header">
            <div className="logo-container">
                <Link to="/">
                <img src={RobotImage} alt="OptiPortfolio Logo" className="logo" />
                </Link>
                <h1>OptiPortfolio</h1>
            </div>
            <nav className="nav-links">
                <Link to="/">Home</Link>
                <Link to="/about">About</Link>
                <Link to="/contact">Contact</Link>
            </nav>
        </header>
    );
}

export default Header;
