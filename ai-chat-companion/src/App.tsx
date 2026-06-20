import { useEffect, useState } from "react";
import { Sidebar } from "@/components/chat/Sidebar";
import { ChatView } from "@/components/chat/ChatView";
import { DocumentPreparationView } from "@/components/documents/DocumentPreparationView";
import { useChats } from "@/hooks/useChats";

export type WorkspaceView = "chat" | "documents";

export default function App() {
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
  const [activeView, setActiveView] = useState<WorkspaceView>("chat");

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
        activeView={activeView}
        collapsed={collapsed}
        onToggleCollapsed={() => setCollapsed((value) => !value)}
        onSelect={(id) => {
          setActiveView("chat");
          setActiveId(id);
        }}
        onNew={() => {
          setActiveView("chat");
          newChat();
        }}
        onRename={renameChat}
        onDelete={deleteChat}
        onSelectDocuments={() => setActiveView("documents")}
      />
      {activeView === "chat" ? (
        <ChatView
          chat={activeChat}
          onSend={sendMessage}
          isSending={isSending}
          theme={theme}
          onToggleTheme={() => setTheme((value) => (value === "dark" ? "light" : "dark"))}
        />
      ) : (
        <DocumentPreparationView
          theme={theme}
          onToggleTheme={() => setTheme((value) => (value === "dark" ? "light" : "dark"))}
        />
      )}
    </div>
  );
}
