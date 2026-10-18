"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import { ChatMessage, MODE_LABELS } from "../lib/api";

export default function MessageBubble({ message }: { message: ChatMessage }) {
  const [showSources, setShowSources] = useState(false);
  const isUser = message.role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[75%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
          isUser
            ? "bg-brand-600 text-white rounded-br-sm"
            : "bg-slate-800 text-slate-100 rounded-bl-sm"
        }`}
      >
        <div className="prose prose-invert prose-sm max-w-none prose-p:my-1 prose-ol:my-1 prose-ul:my-1">
          <ReactMarkdown>{message.content}</ReactMarkdown>
        </div>

        {!isUser && message.retrieval_mode && (
          <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px]">
            <span className="rounded-full bg-slate-700 px-2 py-0.5 text-slate-300">
              {MODE_LABELS[message.retrieval_mode] ?? message.retrieval_mode}
            </span>
            {message.sufficiency != null && (
              <span
                className="rounded-full bg-slate-700 px-2 py-0.5 text-slate-300"
                title="How well the retrieved manual sections support an answer"
              >
                evidence {Math.round(message.sufficiency * 100)}%
              </span>
            )}
            {message.abstained && (
              <span className="rounded-full bg-amber-900/60 px-2 py-0.5 text-amber-200">
                Insufficient evidence
              </span>
            )}
          </div>
        )}

        {!isUser && message.sources.length > 0 && (
          <div className="mt-2 pt-2 border-t border-slate-700">
            <button
              onClick={() => setShowSources((s) => !s)}
              className="text-xs text-brand-400 hover:text-brand-300 font-medium"
            >
              {showSources ? "Hide" : message.abstained ? "Show closest" : "Show"} sources ({message.sources.length} page
              {message.sources.length > 1 ? "s" : ""})
            </button>
            {showSources && (
              <div className="mt-2 space-y-2">
                {message.sources.map((s, i) => (
                  <div
                    key={i}
                    className="text-xs bg-slate-900/60 border border-slate-700 rounded-lg p-2"
                  >
                    <span className="font-semibold text-brand-300">Page {s.page}</span>
                    <p className="text-slate-400 mt-1">{s.snippet}</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
