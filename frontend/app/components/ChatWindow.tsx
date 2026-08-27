"use client";

import { useEffect, useRef, useState } from "react";
import { api, ChatMessage, Manual } from "../lib/api";
import MessageBubble from "./MessageBubble";

const SUGGESTED_QUESTIONS = [
  "How do I reset the tire pressure monitoring system?",
  "What type of engine oil should I use?",
  "How do I jump-start the battery?",
  "What do the dashboard warning lights mean?",
];

export default function ChatWindow({
  manual,
  conversationId,
  onConversationStart,
}: {
  manual: Manual | null;
  conversationId: string | null;
  onConversationStart: (id: string) => void;
}) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (conversationId) {
      api.getConversation(conversationId).then((c) => setMessages(c.messages));
    } else {
      setMessages([]);
    }
  }, [conversationId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function send(question: string) {
    if (!manual || !question.trim() || loading) return;
    setError(null);
    setInput("");

    const userMsg: ChatMessage = {
      id: `temp-${Date.now()}`,
      role: "user",
      content: question,
      sources: [],
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    try {
      const res = await api.chat(manual.id, question, conversationId ?? undefined);
      if (!conversationId) onConversationStart(res.conversation_id);
      const assistantMsg: ChatMessage = {
        id: `temp-a-${Date.now()}`,
        role: "assistant",
        content: res.answer,
        sources: res.sources,
        created_at: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (e: any) {
      setError(e.message ?? "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  if (!manual) {
    return (
      <div className="flex-1 flex items-center justify-center text-slate-500 text-sm">
        Upload or select a vehicle manual to start asking questions.
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col h-full">
      <div className="px-6 py-4 border-b border-slate-800">
        <h2 className="text-slate-100 font-semibold">{manual.vehicle_name}</h2>
        <p className="text-xs text-slate-500">{manual.filename}</p>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
        {messages.length === 0 && (
          <div className="space-y-3">
            <p className="text-sm text-slate-400">Try asking:</p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {SUGGESTED_QUESTIONS.map((q) => (
                <button
                  key={q}
                  onClick={() => send(q)}
                  className="text-left text-sm px-3 py-2 rounded-lg bg-slate-900 border border-slate-800 hover:border-brand-600 text-slate-300 transition-colors"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m) => (
          <MessageBubble key={m.id} message={m} />
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="bg-slate-800 rounded-2xl rounded-bl-sm px-4 py-3 text-sm text-slate-400">
              Searching the manual…
            </div>
          </div>
        )}
        {error && <p className="text-xs text-red-400">{error}</p>}
        <div ref={bottomRef} />
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          send(input);
        }}
        className="p-4 border-t border-slate-800 flex gap-2"
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about your vehicle manual…"
          className="flex-1 px-4 py-3 rounded-xl bg-slate-900 border border-slate-700 text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-brand-500"
        />
        <button
          type="submit"
          disabled={loading || !input.trim()}
          className="px-5 py-3 rounded-xl bg-brand-600 hover:bg-brand-700 disabled:opacity-50 text-sm font-medium transition-colors"
        >
          Ask
        </button>
      </form>
    </div>
  );
}
