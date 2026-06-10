import { useCallback, useEffect, useState } from "react";

export type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
};

export type Chat = {
  id: string;
  title: string;
  messages: Message[];
};

const uid = () => Math.random().toString(36).slice(2, 10);
const STORAGE_KEY = "oracles.chat.history.v1";

function getApiBaseUrl() {
  if (typeof window !== "undefined") {
    const apiBaseUrl = new URLSearchParams(window.location.search).get("apiBaseUrl");
    if (apiBaseUrl) return apiBaseUrl.replace(/\/$/, "");
  }

  return import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") ?? "http://127.0.0.1:8000";
}

const API_BASE_URL = getApiBaseUrl();

type PersistedChatState = {
  chats: Chat[];
  activeId: string | null;
};

async function fetchAssistantReply(query: string): Promise<string> {
  const response = await fetch(`${API_BASE_URL}/api/v1/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Chat request failed with HTTP ${response.status}`);
  }

  const payload = (await response.json()) as { response?: string };
  if (!payload.response) {
    throw new Error("The backend returned an empty response.");
  }

  return payload.response;
}

function loadPersistedState(): PersistedChatState {
  if (typeof window === "undefined") return { chats: [], activeId: null };

  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return { chats: [], activeId: null };

    const parsed = JSON.parse(raw) as Partial<PersistedChatState>;
    if (!Array.isArray(parsed.chats)) return { chats: [], activeId: null };

    return {
      chats: parsed.chats,
      activeId: parsed.activeId ?? parsed.chats[0]?.id ?? null,
    };
  } catch {
    return { chats: [], activeId: null };
  }
}

export function useChats() {
  const [initialState] = useState(loadPersistedState);
  const [chats, setChats] = useState<Chat[]>(initialState.chats);
  const [activeId, setActiveId] = useState<string | null>(initialState.activeId);
  const [isSending, setIsSending] = useState(false);

  const activeChat = chats.find((c) => c.id === activeId) ?? null;

  useEffect(() => {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify({ chats, activeId }));
  }, [chats, activeId]);

  const newChat = useCallback(() => {
    const chat: Chat = { id: uid(), title: "New chat", messages: [] };
    setChats((prev) => [chat, ...prev]);
    setActiveId(chat.id);
  }, []);

  const deleteChat = useCallback(
    (id: string) => {
      setChats((prev) => {
        const next = prev.filter((c) => c.id !== id);
        if (activeId === id) setActiveId(next[0]?.id ?? null);
        return next;
      });
    },
    [activeId],
  );

  const renameChat = useCallback((id: string, title: string) => {
    setChats((prev) => prev.map((c) => (c.id === id ? { ...c, title: title || "Untitled" } : c)));
  }, []);

  const sendMessage = useCallback(
    async (content: string) => {
      if (!content.trim()) return;
      const text = content.trim();
      const targetId = activeId ?? uid();

      setChats((prev) => {
        let list = prev;
        if (!activeId) {
          const chat: Chat = { id: targetId, title: text.slice(0, 40), messages: [] };
          list = [chat, ...prev];
        }
        return list.map((c) => {
          if (c.id !== targetId) return c;
          const isFirst = c.messages.length === 0;
          return {
            ...c,
            title: isFirst ? text.slice(0, 40) : c.title,
            messages: [...c.messages, { id: uid(), role: "user", content: text }],
          };
        });
      });

      if (!activeId) setActiveId(targetId);

      setIsSending(true);
      try {
        const reply = await fetchAssistantReply(text);
        setChats((prev) =>
          prev.map((c) =>
            c.id === targetId
              ? {
                  ...c,
                  messages: [...c.messages, { id: uid(), role: "assistant", content: reply }],
                }
              : c,
          ),
        );
      } catch (error) {
        const message = error instanceof Error ? error.message : "Unable to reach the backend.";
        setChats((prev) =>
          prev.map((c) =>
            c.id === targetId
              ? {
                  ...c,
                  messages: [
                    ...c.messages,
                    {
                      id: uid(),
                      role: "assistant",
                      content: `I could not get a response from the local LLM.\n\n${message}`,
                    },
                  ],
                }
              : c,
          ),
        );
      } finally {
        setIsSending(false);
      }
    },
    [activeId],
  );

  return {
    chats,
    activeChat,
    activeId,
    setActiveId,
    newChat,
    deleteChat,
    renameChat,
    sendMessage,
    isSending,
  };
}
