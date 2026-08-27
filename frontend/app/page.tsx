"use client";

import { useEffect, useState } from "react";
import { api, Manual, Conversation } from "./lib/api";
import Sidebar from "./components/Sidebar";
import ChatWindow from "./components/ChatWindow";

export default function Home() {
  const [manuals, setManuals] = useState<Manual[]>([]);
  const [activeManualId, setActiveManualId] = useState<string | null>(null);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);

  useEffect(() => {
    api.listManuals().then((ms) => {
      setManuals(ms);
      if (ms.length > 0) setActiveManualId(ms[0].id);
    });
  }, []);

  useEffect(() => {
    if (activeManualId) {
      api.listConversations(activeManualId).then(setConversations);
      setActiveConversationId(null);
    } else {
      setConversations([]);
    }
  }, [activeManualId]);

  const activeManual = manuals.find((m) => m.id === activeManualId) ?? null;

  function handleManualUploaded(m: Manual) {
    setManuals((prev) => [m, ...prev]);
    setActiveManualId(m.id);
  }

  function handleConversationStart(id: string) {
    setActiveConversationId(id);
    if (activeManualId) {
      api.listConversations(activeManualId).then(setConversations);
    }
  }

  return (
    <main className="flex h-screen w-screen overflow-hidden">
      <Sidebar
        manuals={manuals}
        activeManualId={activeManualId}
        onSelectManual={(id) => setActiveManualId(id)}
        onManualUploaded={handleManualUploaded}
        conversations={conversations}
        activeConversationId={activeConversationId}
        onSelectConversation={(id) => setActiveConversationId(id)}
        onNewConversation={() => setActiveConversationId(null)}
      />
      <ChatWindow
        manual={activeManual}
        conversationId={activeConversationId}
        onConversationStart={handleConversationStart}
      />
    </main>
  );
}
