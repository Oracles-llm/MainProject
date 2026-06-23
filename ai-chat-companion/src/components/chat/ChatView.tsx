import { useEffect, useRef, useState } from "react";
import { ArrowUp, Brain, Check, ChevronDown, Moon, Paperclip, Sparkles, Sun } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { MessageBubble, TypingBubble } from "./MessageBubble";
import { EmptyState } from "./EmptyState";
import type { Chat, ChatMode } from "@/hooks/useChats";

type Props = {
  chat: Chat | null;
  onSend: (text: string, mode?: ChatMode) => Promise<void>;
  isSending: boolean;
  theme: "light" | "dark";
  onToggleTheme: () => void;
};

export function ChatView({ chat, onSend, isSending, theme, onToggleTheme }: Props) {
  const [input, setInput] = useState("");
  const [mode, setMode] = useState<ChatMode>("normal");
  const scrollRef = useRef<HTMLDivElement>(null);
  const lastMessage = chat?.messages.at(-1);
  const showTyping =
    isSending &&
    (!lastMessage ||
      lastMessage.role !== "assistant" ||
      (lastMessage.content.length === 0 && !lastMessage.thinkingSteps?.length));
  const modeLabel = mode === "thinking" ? "Thinking" : "Normal";
  const ModeIcon = mode === "thinking" ? Brain : Sparkles;

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [chat?.messages.length, isSending, lastMessage?.content, lastMessage?.thinkingSteps?.length]);

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || isSending) return;
    const text = input;
    setInput("");
    await onSend(text, mode);
  };

  const isEmpty = !chat || chat.messages.length === 0;

  return (
    <section className="flex h-full flex-1 flex-col bg-background">
      {/* Top bar */}
      <header className="flex h-14 shrink-0 items-center justify-between border-b border-border px-4">
        <h1 className="truncate text-sm font-semibold text-foreground">
          {chat?.title ?? "New chat"}
        </h1>
        <Button
          variant="ghost"
          size="icon"
          onClick={onToggleTheme}
          aria-label="Toggle theme"
          className="h-8 w-8"
        >
          {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </Button>
      </header>

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto">
        {isEmpty ? (
          <EmptyState onPick={(text) => onSend(text, mode)} />
        ) : (
          <div className="mx-auto max-w-3xl px-4 py-6">
            {chat!.messages.map((m) => (
              <MessageBubble key={m.id} message={m} />
            ))}
            {showTyping && <TypingBubble />}
          </div>
        )}
      </div>

      {/* Composer */}
      <div className="shrink-0 border-t border-border bg-background px-4 py-4">
        <form onSubmit={handleSubmit} className="mx-auto max-w-3xl">
          <div className="flex items-end gap-2 rounded-2xl border border-border bg-card px-3 py-2 shadow-sm focus-within:border-ring focus-within:ring-1 focus-within:ring-ring">
            <button
              type="button"
              className="mb-1 rounded-md p-1.5 text-muted-foreground hover:bg-accent hover:text-accent-foreground"
              aria-label="Attach"
            >
              <Paperclip className="h-4 w-4" />
            </button>
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit();
                }
              }}
              placeholder="Message your assistant..."
              rows={1}
              className="max-h-40 flex-1 resize-none border-0 bg-transparent px-1 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none"
            />
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button
                  type="button"
                  disabled={isSending}
                  className="mb-1 inline-flex h-8 shrink-0 items-center gap-1.5 rounded-full px-2.5 text-sm font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground disabled:pointer-events-none disabled:opacity-50"
                  aria-label="Select answer mode"
                >
                  <ModeIcon className="h-4 w-4" />
                  <span className="hidden sm:inline">{modeLabel}</span>
                  <ChevronDown className="h-3.5 w-3.5" />
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" side="top" className="w-44 rounded-xl p-1.5">
                <DropdownMenuItem
                  onClick={() => setMode("normal")}
                  className="h-9 cursor-pointer justify-between rounded-lg"
                >
                  <span className="inline-flex items-center gap-2">
                    <Sparkles className="h-4 w-4" />
                    Normal
                  </span>
                  {mode === "normal" && <Check className="h-4 w-4" />}
                </DropdownMenuItem>
                <DropdownMenuItem
                  onClick={() => setMode("thinking")}
                  className="h-9 cursor-pointer justify-between rounded-lg"
                >
                  <span className="inline-flex items-center gap-2">
                    <Brain className="h-4 w-4" />
                    Thinking
                  </span>
                  {mode === "thinking" && <Check className="h-4 w-4" />}
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
            <Button
              type="submit"
              size="icon"
              disabled={!input.trim() || isSending}
              className="mb-1 h-8 w-8 rounded-full"
              aria-label="Send"
            >
              <ArrowUp className="h-4 w-4" />
            </Button>
          </div>
          <p className="mt-2 text-center text-xs text-muted-foreground">
            AI can make mistakes. Verify important information.
          </p>
        </form>
      </div>
    </section>
  );
}
