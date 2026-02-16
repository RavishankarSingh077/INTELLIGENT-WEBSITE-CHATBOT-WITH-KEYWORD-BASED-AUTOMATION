/**
 * Project Management Frontend Integration
 * Zero-impact integration with existing chatbot
 */

class ProjectManagement {
    constructor() {
        this.isEnabled = false;
        this.chatbotContainer = null;
        this.chatInterface = null;
        this.init();
    }

    init() {
        // Wait for chatbot to load
        this.waitForChatbot();
    }

    waitForChatbot() {
        const checkChatbot = () => {
            this.chatbotContainer = document.querySelector('.chatbot-container') || 
                                  document.querySelector('#chatbot-container') ||
                                  document.querySelector('.chat-widget');
            
            if (this.chatbotContainer) {
                this.setupProjectManagement();
            } else {
                // Retry after 500ms
                setTimeout(checkChatbot, 500);
            }
        };
        
        checkChatbot();
    }

    setupProjectManagement() {
        try {
            // Add event listeners for button clicks
            this.chatbotContainer.addEventListener('click', (event) => {
                if (event.target.classList.contains('project-button')) {
                    event.preventDefault();
                    this.handleButtonClick(event.target);
                }
            });

            // Listen for messages from popup windows
            window.addEventListener('message', (event) => {
                this.handlePopupMessage(event);
            });

            console.log('Project management system initialized');
        } catch (error) {
            console.error('Error setting up project management:', error);
        }
    }

    handleButtonClick(button) {
        const action = button.dataset.action;
        const additionalData = button.dataset.data || '';

        if (!action) {
            console.error('Button action not defined');
            return;
        }

        console.log(`Handling button action: ${action}`);

        // Send button action to backend
        this.sendButtonAction(action, additionalData);
    }

    async sendButtonAction(action, additionalData = '') {
        try {
            const response = await fetch('/project-action', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    action: action,
                    additional_data: additionalData
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();

            if (result.error) {
                console.error('Project action error:', result.error);
                this.addBotMessage('Sorry, there was an error processing your request. Please try again.');
                return;
            }

            // Handle the response
            this.handleProjectResponse(result);

        } catch (error) {
            console.error('Error sending button action:', error);
            this.addBotMessage('Sorry, there was an error processing your request. Please try again.');
        }
    }

    handleProjectResponse(result) {
        // Add bot response to chat
        if (result.reply) {
            this.addBotMessage(result.reply);
        }

        // Handle form URL if provided
        if (result.form_url) {
            this.openFormPopup(result.form_url, result.form_type);
        }

        // Handle buttons if provided
        if (result.buttons && result.buttons.length > 0) {
            this.addButtonsToChat(result.buttons);
        }
    }

    openFormPopup(formUrl, formType) {
        try {
            // Get chatbot widget dimensions
            const chatbotRect = this.chatbotContainer.getBoundingClientRect();
            
            // Calculate popup position and size
            const popupWidth = Math.max(400, chatbotRect.width);
            const popupHeight = Math.max(600, chatbotRect.height);
            const popupLeft = chatbotRect.left;
            const popupTop = chatbotRect.top;

            // Create popup window
            const popup = window.open(
                formUrl,
                'formPopup',
                `width=${popupWidth},height=${popupHeight},left=${popupLeft},top=${popupTop},scrollbars=yes,resizable=yes,toolbar=no,menubar=no`
            );

            if (popup) {
                // Store form type for later reference
                popup.formType = formType;
                
                // Add form submission handler to popup
                popup.addEventListener('beforeunload', () => {
                    this.handleFormSubmission(formType);
                });

                console.log(`Form popup opened: ${formUrl}`);
            } else {
                console.error('Failed to open popup - popup blocked?');
                this.addBotMessage('Please allow popups to use the form feature.');
            }

        } catch (error) {
            console.error('Error opening form popup:', error);
            this.addBotMessage('Sorry, there was an error opening the form. Please try again.');
        }
    }

    handlePopupMessage(event) {
        // Verify message origin for security
        if (event.origin !== window.location.origin) {
            return;
        }

        if (event.data.type === 'form_submitted') {
            this.handleFormSubmission(event.data.formType);
        }
    }

    async handleFormSubmission(formType) {
        try {
            const response = await fetch('/form-submission', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    form_type: formType
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();

            if (result.error) {
                console.error('Form submission error:', result.error);
                this.addBotMessage('Thank you for your submission. We\'ll get back to you soon.');
                return;
            }

            // Add bot response to chat
            if (result.reply) {
                this.addBotMessage(result.reply);
            }

        } catch (error) {
            console.error('Error handling form submission:', error);
            this.addBotMessage('Thank you for your submission. We\'ll get back to you soon.');
        }
    }

    addBotMessage(message) {
        // Find chat interface and add message
        const chatMessages = document.querySelector('.chat-messages') || 
                           document.querySelector('.messages') ||
                           document.querySelector('#messages');
        
        if (chatMessages) {
            const messageElement = document.createElement('div');
            messageElement.className = 'bot-message message';
            messageElement.innerHTML = `<div class="message-content">${message}</div>`;
            
            chatMessages.appendChild(messageElement);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        } else {
            // Fallback: use console or alert
            console.log('Bot message:', message);
        }
    }

    addButtonsToChat(buttons) {
        // Find chat interface
        const chatMessages = document.querySelector('.chat-messages') || 
                           document.querySelector('.messages') ||
                           document.querySelector('#messages');
        
        if (chatMessages) {
            // Remove existing buttons
            const existingButtons = chatMessages.querySelector('.project-buttons');
            if (existingButtons) {
                existingButtons.remove();
            }

            // Create button container
            const buttonContainer = document.createElement('div');
            buttonContainer.className = 'project-buttons message-buttons';
            
            // Add buttons
            buttons.forEach(button => {
                const buttonElement = document.createElement('button');
                buttonElement.className = 'project-button chat-button';
                buttonElement.textContent = button.text;
                buttonElement.dataset.action = button.action;
                
                if (button.data) {
                    buttonElement.dataset.data = button.data;
                }
                
                buttonContainer.appendChild(buttonElement);
            });

            // Add to chat
            chatMessages.appendChild(buttonContainer);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    }

    // Method to enable project features (can be called externally)
    enable() {
        this.isEnabled = true;
        console.log('Project management features enabled');
    }

    // Method to disable project features
    disable() {
        this.isEnabled = false;
        console.log('Project management features disabled');
    }
}

// Initialize project management system
const projectManagement = new ProjectManagement();

// Export for external use
window.ProjectManagement = projectManagement;

// Auto-enable if project features are available
document.addEventListener('DOMContentLoaded', () => {
    // Check if project features are enabled on the backend
    fetch('/project-action', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'check_status' })
    })
    .then(response => response.json())
    .then(data => {
        if (!data.error) {
            projectManagement.enable();
        }
    })
    .catch(error => {
        console.log('Project features not available:', error);
    });
});


