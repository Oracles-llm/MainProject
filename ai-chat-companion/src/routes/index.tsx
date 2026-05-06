import { useEffect, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { Sidebar } from "@/components/chat/Sidebar";
import { ChatView } from "@/components/chat/ChatView";
import { useChats } from "@/hooks/useChats";

export const Route = createFileRoute("/")({
  component: Index,
  head: () => ({
    meta: [
      { title: "Lumen — AI Chat Assistant" },
      {
        name: "description",
        content:
          "A clean, modern desktop chat interface for AI conversations with sidebar history.",
      },
    ],
  }),
});

function Index() {
  const {
    chats,
    activeChat,
    activeId,
    setActiveId,
    newChat,
    deleteChat,
    renameChat,
    sendMessage,
    isSending,
  } = useChats();

  const [collapsed, setCollapsed] = useState(false);
  const [theme, setTheme] = useState<"light" | "dark">("light");

  useEffect(() => {
    const root = document.documentElement;
    if (theme === "dark") root.classList.add("dark");
    else root.classList.remove("dark");
  }, [theme]);

  return (
    <div className="flex h-screen w-full overflow-hidden bg-background">
      <Sidebar
        chats={chats}
        activeId={activeId}
        collapsed={collapsed}
        onToggleCollapsed={() => setCollapsed((v) => !v)}
        onSelect={setActiveId}
        onNew={newChat}
        onRename={renameChat}
        onDelete={deleteChat}
      />
      <ChatView
        chat={activeChat}
        onSend={sendMessage}
        isSending={isSending}
        theme={theme}
        onToggleTheme={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
      />
    </div>
  );
}
