import "./bootstrap";

// Initialize chat functionality when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
    const chatForm = document.getElementById("chat-form");
    const messageInput = document.getElementById("message-input");
    const messagesContainer = document.getElementById("chat-messages");

    if (chatForm) {
        chatForm.addEventListener("submit", handleChatSubmit);
    }

    async function handleChatSubmit(e) {
        e.preventDefault();
        const message = messageInput.value.trim();
        if (!message) return;

        // Clear input
        messageInput.value = "";

        // Add user message
        appendMessage("user", message);

        try {
            const response = await fetch("/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "X-CSRF-TOKEN": document.querySelector(
                        'meta[name="csrf-token"]'
                    ).content,
                },
                body: JSON.stringify({ message }),
            });

            const data = await response.json();

            if (data.success) {
                appendMessage("bot", data.message);

                // Play audio if available
                if (data.audio_url) {
                    const audio = new Audio(data.audio_url);
                    await audio.play();
                }
            } else {
                appendMessage(
                    "error",
                    "Maaf terjadi kesalahan Silakan coba lagi"
                );
            }
        } catch (error) {
            console.error("Chat error:", error);
            appendMessage("error", "Maaf terjadi kesalahan sistem");
        }
    }

    function appendMessage(type, content) {
        const messageDiv = document.createElement("div");
        messageDiv.className = `p-4 rounded-lg ${
            type === "user" ? "bg-blue-600 ml-auto" : "bg-gray-700"
        } ${type === "error" ? "bg-red-600" : ""} max-w-[80%]`;
        messageDiv.textContent = content;

        const container = document.createElement("div");
        container.className =
            type === "user" ? "flex justify-end" : "flex justify-start";
        container.appendChild(messageDiv);

        const wrapper = messagesContainer.querySelector(".max-w-3xl");
        wrapper.appendChild(container);

        // Scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
});
