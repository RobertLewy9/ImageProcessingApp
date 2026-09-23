
import React, { useState } from 'react';
import './Login.css'; // 确保引入了正确的样式文件

function Login({ onLoginSuccess, onRegisterClick }) {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');

    const handleLogin = async (e) => {
        e.preventDefault();
        setError(''); // Clear previous errors

        try {
            const response = await fetch('http://127.0.0.1:5000/api/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ username, password }),
            });

            if (response.ok) {
                const data = await response.json();
                localStorage.setItem('token', data.token); // Store the token
                onLoginSuccess();
            } else {
                const errorData = await response.json();
                setError(errorData.message || 'Login failed');
            }
        } catch (err) {
            setError('Network error or server is unreachable');
            console.error('Login error:', err);
        }
    };

    return (
        <div className="auth-container"> {/* Changed from login-container to auth-container */}
            <form className="login-form" onSubmit={handleLogin}>
                {/* 替换 h2 标题为 Logo 图片 */}
                <img src="/BG/logo.png" alt="System Logo" className="login-logo" />
                {error && <div className="error-message">{error}</div>}
                <div className="input-group">
                    <label htmlFor="username">用户名</label>
                    <input
                        type="text"
                        id="username"
                        value={username}
                        onChange={(e) => setUsername(e.target.value)}
                        required
                    />
                </div>
                <div className="input-group">
                    <label htmlFor="password">密码</label>
                    <input
                        type="password"
                        id="password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        required
                    />
                </div>
                <button type="submit" className="login-button">登录</button>
                <div className="register-link-container">
                    还没有账号？ <a href="#" onClick={onRegisterClick}>立即注册</a>
                </div>
            </form>
        </div>
    );
}

export default Login;
