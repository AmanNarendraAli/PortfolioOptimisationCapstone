import React from 'react';
import './Footer.css';

function Footer() {
    return (
        <footer className="footer">
            <p className='footerLeftText'>&copy; 2023 OptiPortfolio. All rights reserved.</p>
            <div className="footer-links">
                <a href="#terms">Terms of Service</a>
            </div>
        </footer>
    );
}

export default Footer;
