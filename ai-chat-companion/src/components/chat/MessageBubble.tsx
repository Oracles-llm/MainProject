import { Brain, CheckCircle2, Sparkles, User } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Message } from "@/hooks/useChats";

export function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";
  const thinkingSteps = !isUser ? (message.thinkingSteps ?? []) : [];

  if (!isUser && !message.content && thinkingSteps.length === 0) {
    return null;
  }

  return (
    <div className={cn("flex w-full gap-4 py-5", isUser ? "justify-end" : "justify-start")}>
      {!isUser && (
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground">
          <Sparkles className="h-4 w-4" />
        </div>
      )}
      <div
        className={cn(
          "max-w-[75%] whitespace-pre-wrap rounded-2xl px-4 py-3 text-sm leading-relaxed",
          isUser
            ? "bg-secondary text-secondary-foreground rounded-br-sm"
            : "bg-muted/40 text-foreground rounded-bl-sm",
        )}
      >
        {thinkingSteps.length > 0 && (
          <div className={cn("space-y-1.5", message.content && "mb-3 border-b border-border pb-3")}>
            <div className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
              <Brain className="h-3.5 w-3.5" />
              Thinking
            </div>
            <div className="space-y-1">
              {thinkingSteps.map((step) => (
                <div key={step} className="flex items-start gap-2 text-xs text-muted-foreground">
                  <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                  <span>{step}</span>
                </div>
              ))}
            </div>
          </div>
        )}
        {message.content}
      </div>
      {isUser && (
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-muted text-muted-foreground">
          <User className="h-4 w-4" />
        </div>
      )}
    </div>
  );
}

export function TypingBubble() {
  return (
    <div className="flex w-full gap-4 py-5">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground">
        <Sparkles className="h-4 w-4" />
      </div>
      <div className="flex items-center gap-1 rounded-2xl rounded-bl-sm bg-muted/40 px-4 py-4">
        <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground/60 [animation-delay:-0.3s]" />
        <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground/60 [animation-delay:-0.15s]" />
        <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground/60" />
      </div>
    </div>
  );
}
