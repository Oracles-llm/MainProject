import { useCallback, useEffect, useState } from "react";

export type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  thinkingSteps?: string[];
};

export type ChatMode = "normal" | "thinking";

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

type StreamEvent =
  | { type: "token"; content?: string }
  | { type: "step"; message?: string }
  | { type: "done" }
  | { type: "error"; message?: string };

async function streamAssistantReply(
  query: string,
  mode: ChatMode,
  onTokenContent: (content: string) => void,
  onStep: (message: string) => void,
): Promise<string> {
  const response = await fetch(`${API_BASE_URL}/api/v1/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, mode }),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Chat request failed with HTTP ${response.status}`);
  }

  if (!response.body) {
    throw new Error("The backend did not return a streaming response.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let fullContent = "";
  let pending = "";

  const handleEvent = (event: StreamEvent) => {
    if (event.type === "token") {
      const content = event.content ?? "";
      if (!content) return;
      fullContent += content;
      onTokenContent(fullContent);
      return;
    }

    if (event.type === "step") {
      const message = event.message?.trim();
      if (message) onStep(message);
      return;
    }

    if (event.type === "error") {
      throw new Error(event.message || "The backend reported a streaming error.");
    }
  };

  const handleLine = (line: string) => {
    const trimmed = line.trim();
    if (!trimmed) return;

    try {
      handleEvent(JSON.parse(trimmed) as StreamEvent);
    } catch (error) {
      if (error instanceof SyntaxError) {
        fullContent += line;
        onTokenContent(fullContent);
        return;
      }
      throw error;
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value, { stream: true });
    if (!chunk) continue;

    pending += chunk;
    const lines = pending.split("\n");
    pending = lines.pop() ?? "";
    lines.forEach(handleLine);
  }

  const finalChunk = decoder.decode();
  if (finalChunk) {
    pending += finalChunk;
  }

  if (pending.trim()) {
    handleLine(pending);
  }

  if (!fullContent) {
    throw new Error("The backend returned an empty response.");
  }

  return fullContent;
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
    async (content: string, mode: ChatMode = "normal") => {
      if (!content.trim()) return;
      const text = content.trim();
      const targetId = activeId ?? uid();
      const assistantMessageId = uid();

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
            messages: [
              ...c.messages,
              { id: uid(), role: "user", content: text },
              {
                id: assistantMessageId,
                role: "assistant",
                content: "",
                thinkingSteps: mode === "thinking" ? [] : undefined,
              },
            ],
          };
        });
      });

      if (!activeId) setActiveId(targetId);

      setIsSending(true);
      try {
        await streamAssistantReply(
          text,
          mode,
          (reply) => {
            setChats((prev) =>
              prev.map((c) =>
                c.id === targetId
                  ? {
                      ...c,
                      messages: c.messages.map((message) =>
                        message.id === assistantMessageId
                          ? { ...message, content: reply }
                          : message,
                      ),
                    }
                  : c,
              ),
            );
          },
          (step) => {
            setChats((prev) =>
              prev.map((c) =>
                c.id === targetId
                  ? {
                      ...c,
                      messages: c.messages.map((message) => {
                        if (message.id !== assistantMessageId) return message;
                        const thinkingSteps = message.thinkingSteps ?? [];
                        if (thinkingSteps.includes(step)) return message;
                        return { ...message, thinkingSteps: [...thinkingSteps, step] };
                      }),
                    }
                  : c,
              ),
            );
          },
        );
      } catch (error) {
        const message = error instanceof Error ? error.message : "Unable to reach the backend.";
        setChats((prev) =>
          prev.map((c) =>
            c.id === targetId
              ? {
                  ...c,
                  messages: c.messages.map((chatMessage) =>
                    chatMessage.id === assistantMessageId
                      ? {
                          ...chatMessage,
                          content: `I could not get a response from the local LLM.\n\n${message}`,
                        }
                      : chatMessage,
                  ),
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
