import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import "./App.css";
import HomePage from './components/Homepage/Homepage';
import Header from './components/Header/Header';
import Footer from './components/Footer/Footer';
function App(){
  return(
    <Router>
      <div className='app'>
      <Header />
      <Routes>
        <Route path="/" element={<HomePage />} />
      </Routes>
      <Footer />
      </div>
    </Router>
  )
}

export default App;
