@extends('layouts.app')

@section('content')
    <div class="min-h-screen flex flex-col">
        <!-- Sidebar -->
        <div class="fixed left-0 top-0 h-full w-64 bg-gray-800 p-4 border-r border-gray-700">
            <div class="mb-4">
                <h2 class="text-xl font-bold">DramLearning Chat</h2>
            </div>
            <button class="w-full bg-gray-700 hover:bg-gray-600 text-white font-semibold py-2 px-4 rounded">
                Chat Baru
            </button>
        </div>

        <!-- Main Content -->
        <div class="ml-64 flex-1 flex flex-col">
            <!-- Chat Messages -->
            <div class="flex-1 overflow-y-auto p-4" id="chat-messages">
                <div class="max-w-3xl mx-auto space-y-4">
                    <!-- Welcome Message -->
                    <div class="flex justify-start">
                        <div class="bg-gray-700 p-4 rounded-lg max-w-[80%]">
                            Halo saya asisten AI ada yang bisa saya bantu?
                        </div>
                    </div>
                </div>
            </div>

            <!-- Input Area -->
            <div class="border-t border-gray-700 p-4">
                <div class="max-w-3xl mx-auto">
                    <form id="chat-form" class="flex gap-2">
                        @csrf
                        <input type="text" id="message-input"
                            class="flex-1 bg-gray-700 text-white rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                            placeholder="Ketik pesan Anda di sini..." required>
                        <button type="submit"
                            class="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500">
                            Kirim
                        </button>
                    </form>
                </div>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('chat-form').addEventListener('submit', async function(e) {
            e.preventDefault();

            const messageInput = document.getElementById('message-input');
            const message = messageInput.value;
            if (!message) return;

            // Clear input
            messageInput.value = '';

            // Add user message to chat
            appendMessage('user', message);

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-CSRF-TOKEN': document.querySelector('input[name="_token"]').value
                    },
                    body: JSON.stringify({
                        message
                    })
                });

                const data = await response.json();

                if (data.success) {
                    appendMessage('bot', data.message);

                    // Play audio if available
                    if (data.audio_url) {
                        const audio = new Audio(data.audio_url);
                        audio.play();
                    }
                } else {
                    appendMessage('error', 'Maaf terjadi kesalahan Silakan coba lagi');
                }
            } catch (error) {
                appendMessage('error', 'Maaf terjadi kesalahan sistem');
            }
        });

        function appendMessage(type, content) {
            const messagesDiv = document.getElementById('chat-messages');
            const messageDiv = document.createElement('div');
            messageDiv.className =
                `p-4 rounded-lg ${type === 'user' ? 'bg-blue-600 ml-auto' : 'bg-gray-700'} ${type === 'error' ? 'bg-red-600' : ''} max-w-[80%]`;
            messageDiv.textContent = content;

            const container = document.createElement('div');
            container.className = 'flex justify-end';
            if (type !== 'user') {
                container.className = 'flex justify-start';
            }
            container.appendChild(messageDiv);

            const wrapper = messagesDiv.querySelector('.max-w-3xl');
            wrapper.appendChild(container);

            // Scroll to bottom
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
    </script>
@endsection
