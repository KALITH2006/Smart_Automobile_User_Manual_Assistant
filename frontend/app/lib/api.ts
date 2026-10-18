const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type Manual = {
  id: string;
  filename: string;
  vehicle_name: string;
  page_count: number;
  chunk_count: number;
  created_at: string;
};

export type Source = {
  page: number;
  snippet: string;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources: Source[];
  created_at: string;
  // Adaptive retrieval diagnostics; only present on freshly generated assistant replies
  retrieval_mode?: string | null;
  sufficiency?: number | null;
  abstained?: boolean;
};

export type ChatResponse = {
  conversation_id: string;
  answer: string;
  sources: Source[];
  retrieval_mode?: string | null;
  sufficiency?: number | null;
  abstained?: boolean;
};

export const MODE_LABELS: Record<string, string> = {
  M0: "Exact code lookup",
  M1: "Keyword search",
  M2: "Hybrid search",
  M3: "Iterative search",
};

export type Conversation = {
  id: string;
  manual_id: string;
  title: string;
  created_at: string;
};

export type ConversationDetail = Conversation & { messages: ChatMessage[] };

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API error ${res.status}: ${body}`);
  }
  return res.json();
}

export const api = {
  health: () => fetch(`${API_URL}/api/health`).then((r) => handle(r)),

  listManuals: (): Promise<Manual[]> =>
    fetch(`${API_URL}/api/manuals`).then((r) => handle(r)),

  uploadManual: (file: File, vehicleName: string): Promise<Manual> => {
    const form = new FormData();
    form.append("file", file);
    form.append("vehicle_name", vehicleName);
    return fetch(`${API_URL}/api/manuals/upload`, {
      method: "POST",
      body: form,
    }).then((r) => handle(r));
  },

  deleteManual: (id: string) =>
    fetch(`${API_URL}/api/manuals/${id}`, { method: "DELETE" }).then((r) => handle(r)),

  listConversations: (manualId?: string): Promise<Conversation[]> => {
    const qs = manualId ? `?manual_id=${manualId}` : "";
    return fetch(`${API_URL}/api/conversations${qs}`).then((r) => handle(r));
  },

  getConversation: (id: string): Promise<ConversationDetail> =>
    fetch(`${API_URL}/api/conversations/${id}`).then((r) => handle(r)),

  deleteConversation: (id: string) =>
    fetch(`${API_URL}/api/conversations/${id}`, { method: "DELETE" }).then((r) => handle(r)),

  chat: (
    manualId: string,
    question: string,
    conversationId?: string
  ): Promise<ChatResponse> =>
    fetch(`${API_URL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        manual_id: manualId,
        question,
        conversation_id: conversationId ?? null,
      }),
    }).then((r) => handle(r)),
};
