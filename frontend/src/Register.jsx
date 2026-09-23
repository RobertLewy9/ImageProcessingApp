
import React, { useState } from 'react';
import './Login.css'; // Re-use Login.css for consistent styling

function Register({ onLoginClick }) {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');

    const handleRegister = async (e) => {
        e.preventDefault();
        setError('');
        setSuccess('');

        if (password !== confirmPassword) {
            setError('Passwords do not match!');
            return;
        }

        try {
            const response = await fetch('http://127.0.0.1:5000/api/register', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ username, password }),
            });

            if (response.ok) {
                setSuccess('Registration successful! You can now log in.');
                setUsername('');
                setPassword('');
                setConfirmPassword('');
                // Optionally, redirect to login after a short delay
                // setTimeout(onLoginClick, 2000);
            } else {
                const errorData = await response.json();
                setError(errorData.message || 'Registration failed');
            }
        } catch (err) {
            setError('Network error or server is unreachable');
            console.error('Registration error:', err);
        }
    };

    return (
        <div className="auth-container"> {/* Re-use auth-container for consistent styling */}
            <form className="register-form" onSubmit={handleRegister}>
                <h2>用户注册</h2>
                {error && <div className="error-message">{error}</div>}
                {success && <div className="success-message" style={{ color: '#28a745', marginBottom: '22px' }}>{success}</div>}
                <div className="input-group">
                    <label htmlFor="reg-username">用户名</label>
                    <input
                        type="text"
                        id="reg-username"
                        value={username}
                        onChange={(e) => setUsername(e.target.value)}
                        required
                    />
                </div>
                <div className="input-group">
                    <label htmlFor="reg-password">密码</label>
                    <input
                        type="password"
                        id="reg-password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        required
                    />
                </div>
                <div className="input-group">
                    <label htmlFor="confirm-password">确认密码</label>
                    <input
                        type="password"
                        id="confirm-password"
                        value={confirmPassword}
                        onChange={(e) => setConfirmPassword(e.target.value)}
                        required
                    />
                </div>
                <button type="submit" className="login-button">注册</button>
                <div className="register-link-container">
                    已有账号？ <a href="#" onClick={onLoginClick}>返回登录</a>
                </div>
            </form>
        </div>
    );
}

export default Register;
