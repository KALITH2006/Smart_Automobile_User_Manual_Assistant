"use client";

import { Manual, Conversation } from "../lib/api";
import FileUpload from "./FileUpload";

export default function Sidebar({
  manuals,
  activeManualId,
  onSelectManual,
  onManualUploaded,
  conversations,
  activeConversationId,
  onSelectConversation,
  onNewConversation,
}: {
  manuals: Manual[];
  activeManualId: string | null;
  onSelectManual: (id: string) => void;
  onManualUploaded: (m: Manual) => void;
  conversations: Conversation[];
  activeConversationId: string | null;
  onSelectConversation: (id: string) => void;
  onNewConversation: () => void;
}) {
  return (
    <aside className="w-80 shrink-0 h-full border-r border-slate-800 bg-slate-950/60 flex flex-col">
      <div className="p-4 border-b border-slate-800">
        <h1 className="text-lg font-semibold text-slate-100">🚗 Manual Assistant</h1>
        <p className="text-xs text-slate-400 mt-1">RAG-powered vehicle manual Q&A</p>
      </div>

      <div className="p-4 border-b border-slate-800">
        <FileUpload onUploaded={onManualUploaded} />
      </div>

      <div className="p-4 border-b border-slate-800 overflow-y-auto max-h-64">
        <p className="text-xs uppercase tracking-wide text-slate-500 mb-2">Manuals</p>
        <div className="space-y-1">
          {manuals.length === 0 && (
            <p className="text-xs text-slate-500">No manuals uploaded yet.</p>
          )}
          {manuals.map((m) => (
            <button
              key={m.id}
              onClick={() => onSelectManual(m.id)}
              className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors ${
                activeManualId === m.id
                  ? "bg-brand-600/20 border border-brand-600 text-brand-100"
                  : "hover:bg-slate-800 border border-transparent text-slate-300"
              }`}
            >
              <div className="font-medium truncate">{m.vehicle_name}</div>
              <div className="text-[11px] text-slate-500">
                {m.page_count} pages · {m.chunk_count} chunks
              </div>
            </button>
          ))}
        </div>
      </div>

      <div className="p-4 flex-1 overflow-y-auto">
        <div className="flex items-center justify-between mb-2">
          <p className="text-xs uppercase tracking-wide text-slate-500">Conversations</p>
          {activeManualId && (
            <button
              onClick={onNewConversation}
              className="text-xs text-brand-400 hover:text-brand-300"
            >
              + New
            </button>
          )}
        </div>
        <div className="space-y-1">
          {!activeManualId && (
            <p className="text-xs text-slate-500">Select a manual to see conversations.</p>
          )}
          {activeManualId && conversations.length === 0 && (
            <p className="text-xs text-slate-500">No conversations yet. Ask a question!</p>
          )}
          {conversations.map((c) => (
            <button
              key={c.id}
              onClick={() => onSelectConversation(c.id)}
              className={`w-full text-left px-3 py-2 rounded-lg text-sm truncate transition-colors ${
                activeConversationId === c.id
                  ? "bg-slate-800 text-slate-100"
                  : "hover:bg-slate-800/60 text-slate-400"
              }`}
            >
              {c.title}
            </button>
          ))}
        </div>
      </div>
    </aside>
  );
}
