
import React, { useState, useEffect } from 'react';
import Login from './Login.jsx';
import Register from './Register.jsx';
import './App.css'; // 主功能页面的样式

// 模拟的图片处理函数，实际将通过后端API调用
const imageProcessingFunctions = [
    { name: 'grayscale', label: '灰度化' },
    { name: 'canny', label: '边缘检测 (Canny)' },
    { name: 'sobel', label: '边缘检测 (Sobel)' },
    { name: 'laplacian', label: '边缘检测 (Laplacian)' },
    { name: 'gaussian_blur', label: '高斯模糊' },
    { name: 'sharpen', label: '图像锐化' },
];

// [1] 加入中英文类别映射表
const LABEL_MAP_CN = {
    "Pond": "池塘",
    "Mountain": "山",
    "Park": "公园",
    "Port": "港口",
    "Farmland": "农田",
    "Beach": "沙滩",
    "Desert": "沙漠",
    "RailwayStation": "火车站",
    "Airport": "机场",
    "Unknown Class": "未知类别"
    // 可在此继续补充对应关系
};

function App() {
    const [isLoggedIn, setIsLoggedIn] = useState(false);
    const [showRegister, setShowRegister] = useState(false);
    const [selectedFile, setSelectedFile] = useState(null);
    const [previewImage, setPreviewImage] = useState(null);
    const [processedImage, setProcessedImage] = useState(null);
    const [recognitionResult, setRecognitionResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [uploadHistory, setUploadHistory] = useState([]); // 新增上传历史状态
    const [showChangePasswordForm, setShowChangePasswordForm] = useState(false); // 新增状态来控制密码修改表单的显示
    const [showFunctionMenu, setShowFunctionMenu] = useState(false); // 恢复功能菜单显示状态

    useEffect(() => {
        const token = localStorage.getItem('token');
        if (token) {
            setIsLoggedIn(true);
            fetchUploadHistory(token); // 登录成功后获取历史记录
        }
    }, [isLoggedIn]); // 依赖 isLoggedIn，当登录状态改变时触发

    const fetchUploadHistory = async (token) => {
        try {
            const response = await fetch(`http://127.0.0.1:5000/api/history`, {
                headers: {
                    'x-access-token': token,
                },
            });
            if (response.ok) {
                const data = await response.json();
                setUploadHistory(data);
            } else if (response.status === 401) {
                // Token 无效，强制重新登录
                localStorage.removeItem('token');
                setIsLoggedIn(false);
            } else {
                setError('Failed to fetch upload history.');
            }
        } catch (err) {
            setError('Network error or server is unreachable when fetching history.');
            console.error('Fetch history error:', err);
        }
    };

    const handleLoginSuccess = () => {
        setIsLoggedIn(true);
        setShowRegister(false);
        fetchUploadHistory(localStorage.getItem('token')); // 登录成功后立即获取历史
    };

    const handleLogout = () => {
        localStorage.removeItem('token');
        setIsLoggedIn(false);
        setSelectedFile(null);
        setPreviewImage(null);
        setProcessedImage(null);
        setRecognitionResult(null);
        setError(null);
        setUploadHistory([]); // 注销时清除历史记录
    };

    const handleDeleteHistoryEntry = async (historyId) => {
        if (!window.confirm('确定要删除此历史记录吗？')) {
            return;
        }

        setLoading(true);
        setError(null);
        const token = localStorage.getItem('token');
        if (!token) {
            setError('未授权，请重新登录。');
            setIsLoggedIn(false);
            setLoading(false);
            return;
        }

        try {
            const response = await fetch(`http://127.0.0.1:5000/api/history/${historyId}`, {
                method: 'DELETE',
                headers: {
                    'x-access-token': token,
                },
            });

            if (response.ok) {
                // 删除成功后刷新历史记录
                fetchUploadHistory(token);
            } else if (response.status === 401) {
                setError('Token无效或已过期，请重新登录。');
                setIsLoggedIn(false);
            } else {
                const errorData = await response.json();
                setError(errorData.message || '删除失败。');
            }
        } catch (err) {
            setError('网络错误或服务器无响应。');
            console.error('Delete history error:', err);
        } finally {
            setLoading(false);
        }
    };

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (file) {
            setSelectedFile(file);
            setPreviewImage(URL.createObjectURL(file));
            setProcessedImage(null); // 清除之前的处理图片
            setRecognitionResult(null); // 清除之前的识别结果，当选择新文件时清空所有旧结果
            setError(null);
        }
    };

    const sendImageToBackend = async (operationType) => {
        if (!selectedFile) {
            setError('请先选择一张图片！');
            return;
        }

        setLoading(true);
        setError(null);
        setProcessedImage(null); // 清除之前的处理结果
        setRecognitionResult(null); // 清除之前的识别结果

        const formData = new FormData();
        formData.append('image', selectedFile);
        formData.append('operation', operationType); // For processing, not recognition

        const token = localStorage.getItem('token');
        if (!token) {
            setError('未授权，请重新登录。');
            setIsLoggedIn(false);
            setLoading(false);
            return;
        }

        try {
            const response = await fetch(`http://127.0.0.1:5000/api/process-image`, {
                method: 'POST',
                headers: {
                    'x-access-token': token, // 发送JWT token
                },
                body: formData,
            });

            if (response.ok) {
                const data = await response.json();
                setProcessedImage(`data:image/png;base64,${data.processed_image}`);
            } else if (response.status === 401) {
                setError('Token无效或已过期，请重新登录。');
                setIsLoggedIn(false);
            } else {
                const errorData = await response.json();
                setError(errorData.error || '图片处理失败。');
            }
        } catch (err) {
            setError('网络错误或服务器无响应。');
            console.error('Image processing error:', err);
        } finally {
            setLoading(false);
        }
    };

    const recognizeImage = async () => {
        if (!selectedFile) {
            setError('请先选择一张图片进行识别！');
            return;
        }

        setLoading(true);
        setError(null);
        setRecognitionResult(null); // 清除之前的识别结果
        setProcessedImage(null); // 清除之前的处理结果

        const formData = new FormData();
        formData.append('image', selectedFile);

        const token = localStorage.getItem('token');
        if (!token) {
            setError('未授权，请重新登录。');
            setIsLoggedIn(false);
            setLoading(false);
            return;
        }

        try {
            const response = await fetch(`http://127.0.0.1:5000/api/recognize-image`, {
                method: 'POST',
                headers: {
                    'x-access-token': token, // 发送JWT token
                },
                body: formData,
            });

            if (response.ok) {
                const data = await response.json();
                setRecognitionResult(data.predictions);
            } else if (response.status === 401) {
                setError('Token无效或已过期，请重新登录。');
                setIsLoggedIn(false);
            } else {
                const errorData = await response.json();
                setError(errorData.error || '图像识别失败。');
            }
        } catch (err) {
            setError('网络错误或服务器无响应。');
            console.error('Image recognition error:', err);
        } finally {
            setLoading(false);
        }
    };

    const handleChangePassword = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);
        const token = localStorage.getItem('token');
        if (!token) {
            setError('未授权，请重新登录。');
            setIsLoggedIn(false);
            setLoading(false);
            return;
        }
        
        const oldPassword = e.target.elements.oldPassword.value;
        const newPassword = e.target.elements.newPassword.value;
        
        if (!oldPassword || !newPassword) {
            setError('旧密码和新密码都不能为空。');
            setLoading(false);
            return;
        }

        try {
            const response = await fetch(`http://127.0.0.1:5000/api/change-password`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'x-access-token': token,
                },
                body: JSON.stringify({ old_password: oldPassword, new_password: newPassword }),
            });

            if (response.ok) {
                alert('密码修改成功！请重新登录。');
                handleLogout(); // 密码修改成功后强制注销，让用户重新登录
            } else if (response.status === 401) {
                const errorData = await response.json();
                setError(errorData.message || '旧密码不正确或Token无效。');
            } else {
                const errorData = await response.json();
                setError(errorData.message || '修改密码失败。');
            }
        } catch (err) {
            setError('网络错误或服务器无响应。');
            console.error('Change password error:', err);
        } finally {
            setLoading(false);
        }
    };

    // 一键清除函数
    const handleClearHistory = async () => {
        if (!window.confirm('确定要清空所有历史记录吗？此操作不可恢复！')) return;
        setLoading(true);
        setError(null);
        const token = localStorage.getItem('token');
        try {
            const response = await fetch('http://127.0.0.1:5000/api/history/all', {
                method: 'DELETE',
                headers: { 'x-access-token': token }
            });
            if (response.ok) {
                setUploadHistory([]);
            } else if (response.status === 401) {
                setError('Token无效或已过期，请重新登录。');
                setIsLoggedIn(false);
            } else {
                const errorData = await response.json();
                setError(errorData.message || '清空历史失败。');
            }
        } catch (e) {
            setError('网络错误或服务器无响应。');
        } finally {
            setLoading(false);
        }
    };

    if (!isLoggedIn) {
        // 使用一个独立的 div 来包裹登录/注册组件，并为其设置背景
        return (
            <div className="login-wrapper">
                {showRegister ? <Register onLoginClick={() => setShowRegister(false)} /> : <Login onLoginSuccess={handleLoginSuccess} onRegisterClick={() => setShowRegister(true)} />}
            </div>
        );
    }

    return (
        <div className={`App ${isLoggedIn ? 'logged-in' : 'logged-out'}`}>
            <header className="App-header">
                <div className="header-title-section">
                    <img src="/BG/logo.png" alt="Logo" className="app-logo" /> {/* 新增图标 */}
                    <h1>遥感图像处理与识别系统</h1>
                </div>
                <div className="header-actions">
                    <button onClick={() => alert('更多详情请关注https://github.com/huggingface/pytorch-image-models')} className="about-button">关于</button> {/* 新增“关于”按钮 */}
                    <button onClick={() => setShowChangePasswordForm(true)} className="change-password-button">修改密码</button>
                    <button onClick={handleLogout} className="logout-button">退出登录</button>
                </div>
            </header>
            {showChangePasswordForm && (
                <div className="change-password-overlay">
                    <div className="change-password-form card-element">
                        <h2>修改密码</h2>
                        {error && <p className="error-message">错误: {error}</p>}
                        <form onSubmit={handleChangePassword}>
                            <div className="input-group">
                                <label htmlFor="oldPassword">旧密码:</label>
                                <input type="password" id="oldPassword" name="oldPassword" required />
                            </div>
                            <div className="input-group">
                                <label htmlFor="newPassword">新密码:</label>
                                <input type="password" id="newPassword" name="newPassword" required />
                            </div>
                            <div className="form-actions">
                                <button type="submit" disabled={loading}>提交</button>
                                <button type="button" onClick={() => setShowChangePasswordForm(false)} disabled={loading}>取消</button>
                            </div>
                        </form>
                    </div>
                </div>
            )}
            <main>
                <div className="image-section">
                    <div className="upload-section card-element">
                        <h2>图像上传</h2>
                        <input type="file" id="file-upload" accept="image/*" onChange={handleFileChange} />
                        <label htmlFor="file-upload" className="file-upload-button">
                            选择图片
                        </label>
                        <div className="image-display-area">
                            {previewImage ? (
                                <img src={previewImage} alt="Preview" className="uploaded-image" />
                            ) : (
                                <div className="image-placeholder">待上传图片</div>
                            )}
                        </div>
                    </div>

                    <div className="result-section card-element">
                        <h2>处理结果</h2>
                        {loading && <p className="loading-message">处理中...</p>}
                        {error && <p className="error-message">错误: {error}</p>}
                        <div className="image-display-area">
                            {processedImage && (
                                <img src={processedImage} alt="Processed" className="processed-image" />
                            )}
                            {processedImage && (
                                <button onClick={() => window.open(processedImage, '_blank')} className="download-result-button">下载处理结果</button>
                            )}
                            {recognitionResult && (
                                <div className="recognition-output">
                                    <h3>识别结果:</h3>
                                    {recognitionResult.map((res, index) => (
                                        <p key={index}>
                                          {(LABEL_MAP_CN[res.class] || res.class)}: {(res.probability * 100).toFixed(2)}%
                                        </p>
                                    ))}
                                </div>
                            )}
                            {!loading && !error && !processedImage && !recognitionResult && (
                                <div className="image-placeholder">处理/识别结果</div>
                            )}
                        </div>
                    </div>
                </div>
                <div className="history-section card-element">
                    <h2>上传历史</h2>
                    <button
                        className="clear-history-button"
                        onClick={handleClearHistory}
                        disabled={uploadHistory.length === 0 || loading}
                    >
                        一键清除
                    </button>
                    {uploadHistory.length > 0 ? (
                        <ul className="history-list">
                            {uploadHistory.map((entry) => (
                                <li key={entry.id} className="history-item">
                                    <p>文件名: {entry.original_filename} ({entry.operation_type})</p>
                                    <p>上传时间: {new Date(entry.upload_time).toLocaleString()}</p>
                                    <div className="history-actions">
                                        {entry.original_image_url && (
                                            <a href={`http://127.0.0.1:5000${entry.original_image_url}`} target="_blank" rel="noopener noreferrer">查看原图</a>
                                        )}
                                        {entry.processed_image_url && (
                                            <a href={`http://127.0.0.1:5000${entry.processed_image_url}`} target="_blank" rel="noopener noreferrer">下载处理结果</a>
                                        )}
                                        {entry.heatmap_image_url && (
                                            <a href={`http://127.0.0.1:5000${entry.heatmap_image_url}`} target="_blank" rel="noopener noreferrer">下载热力图</a>
                                        )}
                                        {entry.recognition_file_url && (
                                            <a href={`http://127.0.0.1:5000${entry.recognition_file_url}`} target="_blank" rel="noopener noreferrer">下载识别结果</a>
                                        )}
                                        <button onClick={() => handleDeleteHistoryEntry(entry.id)} className="delete-button">删除</button>
                                    </div>
                                </li>
                            ))}
                        </ul>
                    ) : (
                        <p>暂无上传历史。</p>
                    )}
                </div>
                <div className="controls-section">
                    <div className="function-menu-header" onClick={() => setShowFunctionMenu(!showFunctionMenu)}>
                        <h2>功能选择 {showFunctionMenu ? '◀' : '▶'}</h2> {/* 使用左右箭头表示展开/收起 */}
                    </div>
                    {showFunctionMenu && (
                        <div className="processing-options-horizontal">
                            {imageProcessingFunctions.map((func) => (
                                <button
                                    key={func.name}
                                    onClick={() => sendImageToBackend(func.name)}
                                    disabled={loading || !selectedFile}
                                >
                                    {func.label}
                                </button>
                            ))}
                            <button
                                onClick={recognizeImage}
                                disabled={loading || !selectedFile}
                                className="recognize-button"
                            >
                                识别图像
                            </button>
                        </div>
                    )}
                </div>
            </main>
        </div>
    );
}

export default App;
