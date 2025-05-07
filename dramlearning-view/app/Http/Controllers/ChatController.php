<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Response;

class ChatController extends Controller
{
    public function index()
    {
        return \view('chat.index');
    }

    public function sendMessage(Request $request)
    {
        $question = $request->input('message');

        try {
            $response = Http::post('http://localhost:8000/textbot', [
                'question' => $question
            ]);

            if ($response->successful()) {
                return Response::json([
                    'success' => true,
                    'message' => $response->json('answer'),
                    'audio_url' => $response->json('audio_url')
                ]);
            }

            return Response::json([
                'success' => false,
                'message' => 'Gagal mendapatkan respons dari bot'
            ], 500);
        } catch (\Exception $e) {
            return Response::json([
                'success' => false,
                'message' => 'Terjadi kesalahan sistem'
            ], 500);
        }
    }
}
