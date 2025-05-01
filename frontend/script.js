document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('chat-box');
    const messageInput = document.getElementById('message-input');
    const sendButton = document.getElementById('send-button');
    const resetButton = document.getElementById('reset-button');
    const statusDiv = document.getElementById('status');

    // --- Configuration ---
    // Replace with your *deployed* backend API endpoint URL
    // For local testing with backend running on port 8000:
    // const API_ENDPOINT = 'http://localhost:8000/api/chat';
    // Example for a Render deployment:
    const API_ENDPOINT = 'http://127.0.0.1:8000/api/chat';
    // Function to add a message to the chat box
    function addMessage(text, sender) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', sender); // sender is 'user' or 'assistant' or 'error'
        messageElement.textContent = text;
        chatBox.appendChild(messageElement);
        // Scroll to the bottom
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    // Function to set status message
    function setStatus(message, isError = false) {
        statusDiv.textContent = message;
        statusDiv.style.color = isError ? '#dc3545' : '#666'; // Red for errors
    }

    // Function to send message to backend
    async function sendMessage() {
        const messageText = messageInput.value.trim();
        if (!messageText) return; // Don't send empty messages

        addMessage(messageText, 'user');
        messageInput.value = ''; // Clear input
        setStatus('Sending...'); // Indicate processing

        try {
            const response = await fetch(API_ENDPOINT, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: messageText,
                    reset_context: false // Explicitly false for normal messages
                }),
            });

            if (!response.ok) {
                // Try to get error detail from API response body
                let errorDetail = `HTTP error ${response.status}`;
                try {
                    const errorData = await response.json();
                    errorDetail = errorData.detail || errorDetail; // Use detail if available
                } catch (e) {
                    // Ignore if response body is not JSON or empty
                }
                throw new Error(errorDetail);
            }

            const data = await response.json();
            addMessage(data.response, 'assistant');
            setStatus(''); // Clear status

        } catch (error) {
            console.error('Error sending message:', error);
            addMessage(`Error: ${error.message}`, 'error'); // Display error in chat
            setStatus(`Error: ${error.message}`, true);
        }
    }

    // Function to reset conversation context
    async function resetConversation() {
        setStatus('Resetting context...');
        try {
            const response = await fetch(API_ENDPOINT, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: "", // Message can be empty when resetting
                    reset_context: true
                }),
            });

            if (!response.ok) {
                 let errorDetail = `HTTP error ${response.status}`;
                 try { const errorData = await response.json(); errorDetail = errorData.detail || errorDetail; } catch (e) {}
                throw new Error(`Failed to reset: ${errorDetail}`);
            }

            const data = await response.json();

            // Clear the visual chat box on the frontend
            chatBox.innerHTML = '';
            // Add confirmation and initial greeting
            addMessage(data.response, 'assistant'); // Display confirmation from backend
            addMessage('Hello! How can I help you today?', 'assistant'); // Add a new greeting

            setStatus('Context reset successfully.');
            messageInput.focus(); // Refocus input

        } catch (error) {
            console.error('Error resetting context:', error);
            addMessage(`Error resetting context: ${error.message}`, 'error');
            setStatus(`Error: ${error.message}`, true);
        }
    }


    // Event Listeners
    sendButton.addEventListener('click', sendMessage);
    messageInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });
    resetButton.addEventListener('click', resetConversation);

    // Initial focus
    messageInput.focus();
});