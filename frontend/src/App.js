import React, { useState, useRef, useEffect } from 'react';
import './App.css';

function App() {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Chào bạn! Bạn muốn hỏi gì về các tài liệu?' }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    // Add user message
    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:5000/api/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: input }),
      });

      const data = await response.json();

      if (response.ok) {
        // Add assistant message
        setMessages(prev => [...prev, { 
          role: 'assistant', 
          content: data.answer,
          sources: data.sources 
        }]);
      } else {
        setMessages(prev => [...prev, { 
          role: 'assistant', 
          content: `Lỗi: ${data.error}` 
        }]);
      }
    } catch (error) {
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: `Lỗi kết nối: ${error.message}` 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Hệ thống Hỏi Đáp Tài Liệu</h1>
      </header>
      
      <div className="chat-container">
        {messages.map((message, index) => (
          <div key={index} className={`message ${message.role}-message`}>
            <div className="message-content">{message.content}</div>
            {message.sources && message.sources.length > 0 && (
              <div className="sources">
                <h4>Nguồn tham khảo:</h4>
                {message.sources.map((source, idx) => (
                  <div key={idx} className="source-item">
                    <p>File: {source.file_name}</p>
                    <p>Đường dẫn: {source.file_path}</p>
                    <p>Điểm tương đồng: {typeof source.score === 'number' ? source.score.toFixed(4) : source.score}</p>

                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
        {isLoading && (
          <div className="message assistant-message">
            <div className="loading">Đang tìm câu trả lời...</div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSubmit} className="input-form">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Nhập câu hỏi của bạn..."
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading || !input.trim()}>
          Gửi
        </button>
      </form>
    </div>
  );
}

export default App;