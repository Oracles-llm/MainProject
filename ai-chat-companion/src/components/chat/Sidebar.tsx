import { useState } from "react";
import {
  Plus,
  MessageSquare,
  Pencil,
  Trash2,
  PanelLeftClose,
  PanelLeftOpen,
  Check,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import type { Chat } from "@/hooks/useChats";

type Props = {
  chats: Chat[];
  activeId: string | null;
  collapsed: boolean;
  onToggleCollapsed: () => void;
  onSelect: (id: string) => void;
  onNew: () => void;
  onRename: (id: string, title: string) => void;
  onDelete: (id: string) => void;
};

export function Sidebar({
  chats,
  activeId,
  collapsed,
  onToggleCollapsed,
  onSelect,
  onNew,
  onRename,
  onDelete,
}: Props) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [confirmId, setConfirmId] = useState<string | null>(null);

  const startEdit = (chat: Chat) => {
    setEditingId(chat.id);
    setDraft(chat.title);
  };

  const commitEdit = () => {
    if (editingId) onRename(editingId, draft.trim());
    setEditingId(null);
  };

  return (
    <aside
      className={cn(
        "flex h-full flex-col border-r border-border bg-sidebar text-sidebar-foreground transition-all duration-200",
        collapsed ? "w-[60px]" : "w-[260px]",
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between p-2">
        {!collapsed && <span className="px-2 text-sm font-semibold tracking-tight">Chats</span>}
        <Button
          variant="ghost"
          size="icon"
          onClick={onToggleCollapsed}
          className="h-8 w-8 text-sidebar-foreground hover:bg-sidebar-accent"
          aria-label="Toggle sidebar"
        >
          {collapsed ? (
            <PanelLeftOpen className="h-4 w-4" />
          ) : (
            <PanelLeftClose className="h-4 w-4" />
          )}
        </Button>
      </div>

      {/* New chat */}
      <div className="px-2 pb-2">
        <Button
          onClick={onNew}
          className={cn(
            "w-full justify-start gap-2 bg-sidebar-accent text-sidebar-accent-foreground hover:bg-sidebar-accent/80",
            collapsed && "justify-center px-0",
          )}
          variant="ghost"
        >
          <Plus className="h-4 w-4" />
          {!collapsed && <span className="text-sm font-medium">New chat</span>}
        </Button>
      </div>

      {!collapsed && (
        <div className="px-4 pb-2 pt-1 text-xs font-medium uppercase tracking-wider text-muted-foreground">
          Recent
        </div>
      )}

      {/* List */}
      <div className="flex-1 overflow-y-auto px-2 pb-2">
        {chats.length === 0 && !collapsed && (
          <div className="px-3 py-6 text-center text-xs text-muted-foreground">
            No chats yet. Start a new one.
          </div>
        )}
        <ul className="space-y-1">
          {chats.map((chat) => {
            const active = chat.id === activeId;
            const editing = editingId === chat.id;
            const confirming = confirmId === chat.id;
            return (
              <li key={chat.id}>
                <div
                  className={cn(
                    "group relative flex items-center gap-2 rounded-md px-2 py-2 text-sm transition-colors",
                    active
                      ? "bg-sidebar-accent text-sidebar-accent-foreground"
                      : "text-sidebar-foreground hover:bg-sidebar-accent/60",
                    collapsed && "justify-center",
                  )}
                >
                  <button
                    onClick={() => !editing && onSelect(chat.id)}
                    className={cn(
                      "flex min-w-0 flex-1 items-center gap-2",
                      collapsed && "justify-center",
                    )}
                  >
                    <MessageSquare className="h-4 w-4 shrink-0" />
                    {!collapsed && !editing && (
                      <span className="truncate text-left">{chat.title}</span>
                    )}
                  </button>

                  {!collapsed && editing && (
                    <div className="flex flex-1 items-center gap-1">
                      <input
                        autoFocus
                        value={draft}
                        onChange={(e) => setDraft(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") commitEdit();
                          if (e.key === "Escape") setEditingId(null);
                        }}
                        className="min-w-0 flex-1 rounded-sm border border-border bg-background px-1 text-sm outline-none focus:ring-1 focus:ring-ring"
                      />
                      <button
                        onClick={commitEdit}
                        className="rounded p-1 hover:bg-background/60"
                        aria-label="Save"
                      >
                        <Check className="h-3.5 w-3.5" />
                      </button>
                      <button
                        onClick={() => setEditingId(null)}
                        className="rounded p-1 hover:bg-background/60"
                        aria-label="Cancel"
                      >
                        <X className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  )}

                  {!collapsed && !editing && !confirming && (
                    <div className="ml-auto hidden items-center gap-0.5 group-hover:flex">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          startEdit(chat);
                        }}
                        className="rounded p-1 text-muted-foreground hover:bg-background/60 hover:text-foreground"
                        aria-label="Rename"
                      >
                        <Pencil className="h-3.5 w-3.5" />
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setConfirmId(chat.id);
                        }}
                        className="rounded p-1 text-muted-foreground hover:bg-background/60 hover:text-destructive"
                        aria-label="Delete"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  )}

                  {!collapsed && confirming && (
                    <div className="ml-auto flex items-center gap-1 text-xs">
                      <span className="text-muted-foreground">Delete?</span>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onDelete(chat.id);
                          setConfirmId(null);
                        }}
                        className="rounded px-1.5 py-0.5 text-destructive hover:bg-destructive/10"
                      >
                        Yes
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setConfirmId(null);
                        }}
                        className="rounded px-1.5 py-0.5 hover:bg-background/60"
                      >
                        No
                      </button>
                    </div>
                  )}
                </div>
              </li>
            );
          })}
        </ul>
      </div>
    </aside>
  );
}
